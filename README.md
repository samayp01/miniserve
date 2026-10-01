# miniserve

An LLM inference server built from scratch on MLX to demo some systems like the KV cache, paged attention, continuous batching, preemption, and streaming.

## Setup

```bash
uv sync
```

## Run

Start the server (port 8000):

```bash
uv run python -m src.api.server
```

Send a streaming request:

```bash
curl -N -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a haiku about the ocean.","max_tokens":40}'
```

Run a prompt from the terminal:

```bash
uv run python -m src.api.chat "Explain water boiling in one paragraph."
```

## Test

```bash
uv run pytest
```

## Benchmark

Run the full suite (starts each server, benchmarks it, shuts it down):

```bash
./benchmarks/run_all.sh
```

Benchmark a single running server:

```bash
uv run python -m benchmarks.bench --target miniserve   # or mlx-lm, vllm-metal
```

Arguments passed to `run_all.sh` go through to the bench, e.g. `./benchmarks/run_all.sh --spec legacy`. Workload specs live in [benchmarks/workload.py](benchmarks/workload.py):

- `mixed` (default): 60% short chat questions, 25% medium (200–450 prompt tokens), 15% long (700–1500 prompt tokens, longer than one prefill chunk)
- `legacy`: only the short questions, matching the runs in STATS.md before the workload spec existed

Every prompt starts with a prefix unique to its QPS level and position, so no two requests share more than the system header. Without it, every level would resend the same prompts and servers with a prefix cache (vllm-metal, mlx-lm) would skip prefill, measuring cache hits instead of the engine.

Each run writes JSON to `benchmarks/results/<spec>/<target>-<git sha>.json`. It includes the environment, config, per-level summaries, per-request token timestamps and (for miniserve) server metrics sampled from `GET /metrics`.

Internal microbenchmarks (paging, batching, preemption):

```bash
uv run python -m benchmarks.measure
```

Results and analysis: [STATS.md](STATS.md).
