import argparse
import asyncio
import json
import random
import time

import httpx

PROMPTS = [
    "Name three primary colors.",
    "Explain what a hash map is in two sentences.",
    "Write a short paragraph about the ocean.",
    "List the first five prime numbers and why they're prime.",
    "Summarize how photosynthesis works for a ten-year-old.",
    "Give three tips for writing clean code.",
]

PORTS = {"miniserve": 8000, "mlx-lm": 8081, "vllm-metal": 8080}
MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


async def one_request(client, base, target, prompt, max_tokens):
    t_send = time.perf_counter()
    t_first = None
    delta_count = 0
    reported_tokens = None

    if target == "miniserve":
        path = "/generate"
        payload = {"prompt": prompt, "max_tokens": max_tokens}
    else:
        path = "/v1/chat/completions"
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "stream_options": {"include_usage": True},
        }

    async with client.stream("POST", base + path, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]" or not data:
                continue
            obj = json.loads(data)
            if target == "miniserve":
                if "delta" in obj:
                    if t_first is None:
                        t_first = time.perf_counter()
                    delta_count += 1
                if obj.get("done"):
                    reported_tokens = obj.get("tokens")
            else:
                choices = obj.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    delta_count += 1
                if obj.get("usage"):
                    reported_tokens = obj["usage"].get("completion_tokens")

    t_done = time.perf_counter()
    out_tokens = reported_tokens if reported_tokens is not None else delta_count
    return {"send": t_send, "first": t_first, "done": t_done, "out_tokens": out_tokens}


async def run_load(base, target, qps, num_requests, max_tokens):
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = []
        for _ in range(num_requests):
            prompt = random.choice(PROMPTS)
            tasks.append(asyncio.create_task(
                one_request(client, base, target, prompt, max_tokens)))
            await asyncio.sleep(random.expovariate(qps))
        return await asyncio.gather(*tasks)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round((p / 100) * (len(xs) - 1))))
    return xs[k]


def summarize(results):
    ttfts = [(r["first"] - r["send"]) * 1000 for r in results if r["first"]]
    lats = [(r["done"] - r["send"]) * 1000 for r in results]
    itls = [(r["done"] - r["first"]) * 1000 / (r["out_tokens"] - 1)
            for r in results if r["first"] and r["out_tokens"] > 1]
    total_out = sum(r["out_tokens"] for r in results)
    span = max(r["done"] for r in results) - min(r["send"] for r in results)
    return {
        "throughput": total_out / span if span else float("nan"),
        "ttft_p50": pct(ttfts, 50), "ttft_p99": pct(ttfts, 99),
        "itl_p50": pct(itls, 50),
        "lat_p50": pct(lats, 50), "lat_p99": pct(lats, 99),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["miniserve", "mlx-lm", "vllm-metal"], required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--qps-list", default="1,2,4,8,16,32")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    base = args.url or f"http://127.0.0.1:{PORTS[args.target]}"

    try:
        if args.warmup:
            await run_load(base, args.target, 1000, args.warmup, args.max_tokens)
    except httpx.ConnectError:
        print(f"could not reach {args.target} at {base} — is the server running?")
        return

    print(f"\n{args.target} @ {base}  ({args.num_requests} reqs/level, max_tokens={args.max_tokens})\n")
    print(f"{'qps':>5}  {'tok/s':>7}  {'ttft_p50':>9}  {'ttft_p99':>9}  {'itl_p50':>8}  {'lat_p50':>8}  {'lat_p99':>8}")
    for qps in [float(x) for x in args.qps_list.split(",")]:
        results = await run_load(base, args.target, qps, args.num_requests, args.max_tokens)
        s = summarize(results)
        print(f"{qps:>5g}  {s['throughput']:>7.0f}  {s['ttft_p50']:>7.0f}ms  {s['ttft_p99']:>7.0f}ms  "
              f"{s['itl_p50']:>6.1f}ms  {s['lat_p50']:>6.0f}ms  {s['lat_p99']:>6.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
