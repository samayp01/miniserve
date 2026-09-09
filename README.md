# miniserve

A small LLM inference server built from scratch on [MLX](https://github.com/ml-explore/mlx) (Apple Silicon) to understand how systems like vLLM actually work. Each piece — KV cache, paged attention, continuous batching, preemption, streaming — was implemented and benchmarked in isolation before being wired into a serving stack.

Model: `mlx-community/Llama-3.2-1B-Instruct-4bit`. Hardware: M4 Max, 36GB unified memory.

## What's inside

```
HTTP (FastAPI + SSE)          src/server.py    streaming request/response, async engine loop
  -> Engine                   src/engine.py    scheduler: admission, continuous batching, preemption
    -> paged KV cache         src/cache/       fixed-size blocks in a shared pool + free-list allocator
    -> model execution        src/model_runner.py   prefill + batched decode on MLX
```

A request enters the waiting queue, the engine admits it when blocks are free, prefills it, then advances it one token per decode step alongside every other running request. Tokens stream back over Server-Sent Events as they're produced.

## Setup

```bash
uv sync
```

## Run

Start the server:
```bash
uv run python -m src.server
```

Send a streaming request:
```bash
curl -N -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a haiku about the ocean.","max_tokens":40}'
```

Or drive it from the terminal:
```bash
uv run python -m src.chat "Explain water boiling in one paragraph."
```

## Test

```bash
uv run pytest
```

## Benchmark

Load test against a running server (miniserve on :8000, mlx-lm on :8081, vllm-metal on :8080):
```bash
uv run python -m benchmarks.bench --target miniserve
uv run python -m benchmarks.bench --target mlx-lm
uv run python -m benchmarks.bench --target vllm-metal
```

Internal microbenchmarks (paging concurrency, static vs continuous batching, preemption load curve) that back the numbers in [STATS.md](STATS.md):
```bash
uv run python -m benchmarks.measure
```

## Findings

Sweeping request rate (QPS 1–32) against all three servers on the same box and model, all three show the expected shape: throughput scales roughly linearly until the GPU saturates, then flattens while tail latency climbs.

| server | throughput ceiling | notes |
|--------|-------------------:|-------|
| miniserve  | ~650 tok/s | single-request latency competitive; batched-execution overhead caps throughput |
| mlx-lm     | ~880 tok/s | highest throughput ceiling |
| vllm-metal | ~590 tok/s | lowest/flattest TTFT under load, but higher ITL and lower ceiling |

miniserve's per-request decode latency lands within range of the reference implementations; the throughput gap at high concurrency is the remaining overhead in its batched execution path. Full tables and the QPS-vs-latency/throughput graph are in [STATS.md](STATS.md).

![benchmark results](benchmarks/results.png)

