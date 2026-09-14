#!/usr/bin/env bash
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="mlx-community/Llama-3.2-1B-Instruct-4bit"
LOGDIR="$REPO/benchmarks/logs"
mkdir -p "$LOGDIR"
BENCH_ARGS=("$@")

SERVER_PID=""
SERVER_PATTERN=""
SERVER_PORT=""

port_pids() {
    lsof -ti "tcp:$1" 2>/dev/null
}

stop_server() {
    [ -n "$SERVER_PID$SERVER_PATTERN$SERVER_PORT" ] || return 0
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    [ -n "$SERVER_PATTERN" ] && pkill -f "$SERVER_PATTERN" 2>/dev/null
    if [ -n "$SERVER_PORT" ]; then
        local i
        for ((i = 0; i < 20; i++)); do
            [ -z "$(port_pids "$SERVER_PORT")" ] && break
            sleep 0.5
        done
        local pids
        pids="$(port_pids "$SERVER_PORT")"
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    fi
    SERVER_PID=""
    SERVER_PATTERN=""
    SERVER_PORT=""
}

trap 'stop_server; exit 130' INT TERM
trap stop_server EXIT

wait_ready() {
    local url="$1" i
    for ((i = 0; i < 180; i++)); do
        kill -0 "$SERVER_PID" 2>/dev/null || return 1
        curl -s -o /dev/null -m 2 "$url" && return 0
        sleep 1
    done
    return 1
}

run_target() {
    local target="$1" port="$2" pattern="$3" ready_url="$4"; shift 4
    local start=("$@")

    echo
    echo "=================================================="
    echo "  $target"
    echo "=================================================="

    local stale
    stale="$(port_pids "$port")"
    [ -n "$stale" ] && { echo "clearing stale process on port $port ..."; kill -9 $stale 2>/dev/null; sleep 1; }

    echo "starting server (log: benchmarks/logs/$target.log) ..."
    "${start[@]}" > "$LOGDIR/$target.log" 2>&1 &
    SERVER_PID=$!
    SERVER_PATTERN="$pattern"
    SERVER_PORT="$port"

    if wait_ready "$ready_url"; then
        echo "ready; running bench ..."
        uv run python -m benchmarks.bench --target "$target" "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"
    else
        echo "!! $target never became ready; skipping (see benchmarks/logs/$target.log)"
    fi

    stop_server
}

run_target miniserve  8000 "src.server"    "http://127.0.0.1:8000/openapi.json" \
    uv run python -m src.server
run_target mlx-lm     8081 "mlx_lm.server" "http://127.0.0.1:8081/v1/models" \
    uv run mlx_lm.server --model "$MODEL" --port 8081
run_target vllm-metal 8080 "vllm serve"    "http://127.0.0.1:8080/v1/models" \
    "$HOME/.venv-vllm-metal/bin/vllm" serve "$MODEL" --port 8080

echo
echo "done."
