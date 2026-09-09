# miniserve

An LLM inference server built from scratch on MLX to demo some systems like the KV cache, paged attention, continuous batching, preemption, and streaming.

## Setup

```bash
uv sync
```

## Run

Start the server (port 8000):

```bash
uv run python -m src.server
```

Send a streaming request:

```bash
curl -N -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a haiku about the ocean.","max_tokens":40}'
```

Run a prompt from the terminal:

```bash
uv run python -m src.chat "Explain water boiling in one paragraph."
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

Internal microbenchmarks (paging, batching, preemption):

```bash
uv run python -m benchmarks.measure
```

Results and analysis: [STATS.md](STATS.md).
