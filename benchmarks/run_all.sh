#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
LOGDIR="$REPO/benchmarks/logs"
mkdir -p "$LOGDIR"
BENCH_ARGS=("$@")
SERVER_PID=""

kill_server() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    pkill -f "src.server"    2>/dev/null
    pkill -f "mlx_lm.server" 2>/dev/null
    pkill -f "vllm serve"    2>/dev/null
    SERVER_PID=""
}
trap kill_server EXIT INT TERM

wait_ready() {
    local url="$1" timeout=240 i
    for ((i = 0; i < timeout; i++)); do
        curl -s -o /dev/null -m 2 "$url" && return 0
        sleep 1
    done
    return 1
}

run_target() {
    local target="$1" ready_url="$2"; shift 2
    local start=("$@")

    echo
    echo "=================================================="
    echo "  $target"
    echo "=================================================="
    echo "starting server (log: benchmarks/logs/$target.log) ..."
    "${start[@]}" > "$LOGDIR/$target.log" 2>&1 &
    SERVER_PID=$!

    if wait_ready "$ready_url"; then
        echo "Running benchmarks ..."
        uv run python -m benchmarks.bench --target "$target" "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"
    else
        echo "!! $target never became ready; skipping (see benchmarks/logs/$target.log)"
    fi

    kill_server
    sleep 3
}

run_target miniserve  "http://127.0.0.1:8000/openapi.json" \
    uv run python -m src.server
run_target mlx-lm     "http://127.0.0.1:8081/v1/models" \
    uv run mlx_lm.server --model "$MODEL" --port 8081
run_target vllm-metal "http://127.0.0.1:8080/v1/models" \
    "$HOME/.venv-vllm-metal/bin/vllm" serve "$MODEL" --port 8080

echo
