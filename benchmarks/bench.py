import argparse
import asyncio
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import httpx

from benchmarks.workload import MODEL, SPECS, Workload

PORTS = {"miniserve": 8000, "mlx-lm": 8081, "vllm-metal": 8080}
RESULTS = Path(__file__).parent / "results"


async def one_request(client, base, target, prompt, max_tokens):
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

    t_send = time.perf_counter()
    token_times = []
    reported_tokens = None
    stages = None

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
                    token_times.append(time.perf_counter())
                if obj.get("done"):
                    reported_tokens = obj.get("tokens")
                    stages = {k: obj.get(k) for k in ("queue_ms", "encode_ms", "prefill_ms", "decode_ms")}
            else:
                choices = obj.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    token_times.append(time.perf_counter())
                if obj.get("usage"):
                    reported_tokens = obj["usage"].get("completion_tokens")

    return {
        "send": t_send,
        "token_times": token_times,
        "done": time.perf_counter(),
        "out_tokens": reported_tokens if reported_tokens is not None else len(token_times),
        "stages": stages,
    }


async def run_load(client, base, target, schedule, max_tokens):
    tasks = []
    for req in schedule:
        tasks.append(asyncio.create_task(one_request(client, base, target, req["prompt"], max_tokens)))
        if req["gap"]:
            await asyncio.sleep(req["gap"])
    results = await asyncio.gather(*tasks)
    return [{"bucket": req["bucket"], "prompt_tokens": req["prompt_tokens"], **res}
            for req, res in zip(schedule, results)]


async def sample_metrics(client, base, stop, interval=0.05):
    samples = []
    while not stop.is_set():
        try:
            samples.append((await client.get(base + "/metrics")).json())
        except httpx.HTTPError:
            pass
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except asyncio.TimeoutError:
            pass
    return samples


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round((p / 100) * (len(xs) - 1))))
    return xs[k]


def summarize(results):
    ttfts = [(r["token_times"][0] - r["send"]) * 1000 for r in results if r["token_times"]]
    lats = [(r["done"] - r["send"]) * 1000 for r in results]
    tpots = [(r["done"] - r["token_times"][0]) * 1000 / (r["out_tokens"] - 1)
             for r in results if r["token_times"] and r["out_tokens"] > 1]
    itls = [(b - a) * 1000 for r in results for a, b in zip(r["token_times"], r["token_times"][1:])]
    total_out = sum(r["out_tokens"] for r in results)
    span = max(r["done"] for r in results) - min(r["send"] for r in results)
    by_bucket = {}
    for name in dict.fromkeys(r["bucket"] for r in results):
        bt = [(r["token_times"][0] - r["send"]) * 1000 for r in results if r["bucket"] == name and r["token_times"]]
        by_bucket[name] = {"n": sum(r["bucket"] == name for r in results),
                           "ttft_p50": pct(bt, 50), "ttft_p99": pct(bt, 99)}
    return {
        "throughput": total_out / span if span else float("nan"),
        "ttft_p50": pct(ttfts, 50), "ttft_p99": pct(ttfts, 99),
        "tpot_p50": pct(tpots, 50),
        "itl_p50": pct(itls, 50), "itl_p99": pct(itls, 99), "itl_max": max(itls, default=float("nan")),
        "lat_p50": pct(lats, 50), "lat_p99": pct(lats, 99),
        "by_bucket": by_bucket,
    }


def summarize_server(samples):
    if not samples:
        return None
    return {
        "max_waiting": max(s["waiting"] for s in samples),
        "max_running": max(s["running"] for s in samples),
        "preemptions": samples[-1]["preemptions"] - samples[0]["preemptions"],
        "min_free_blocks": min(s["free_blocks"] for s in samples),
        "total_blocks": samples[0]["total_blocks"],
        "max_active_mb": max(s["active_mb"] for s in samples),
        "peak_mb": samples[-1]["peak_mb"],
    }


def request_record(r):
    return {
        "bucket": r["bucket"],
        "prompt_tokens": r["prompt_tokens"],
        "out_tokens": r["out_tokens"],
        "latency_ms": round((r["done"] - r["send"]) * 1000, 1),
        "stages": r["stages"],
        "token_ms": [round((t - r["send"]) * 1000, 1) for t in r["token_times"]],
    }


def _run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def environment():
    mem = _run("sysctl", "-n", "hw.memsize")
    return {
        "git_sha": _run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_run("git", "status", "--porcelain", "--untracked-files=no")),
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": round(int(mem) / 2**30) if mem else None,
        "os": platform.platform(),
        "mlx": _version("mlx"),
        "mlx_lm": _version("mlx-lm"),
        "model": MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _ms(x):
    return f"{x:>7.0f}ms"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(PORTS), required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--spec", choices=list(SPECS), default="mixed")
    ap.add_argument("--qps-list", default="1,2,4,8,16,32")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="results path (default: benchmarks/results/<spec>/<target>-<sha>.json)")
    args = ap.parse_args()

    base = args.url or f"http://127.0.0.1:{PORTS[args.target]}"
    workload = Workload(args.spec, args.seed)
    env = environment()
    levels = []

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            if args.warmup:
                await run_load(client, base, args.target, workload.schedule(1000, args.warmup, "warmup"), args.max_tokens)
        except httpx.ConnectError:
            print(f"could not reach {args.target} at {base} — is the server running?")
            return
        has_metrics = args.target == "miniserve"

        print(f"\n{args.target} @ {base}  spec={args.spec} seed={args.seed}  "
              f"({args.num_requests} reqs/level, max_tokens={args.max_tokens})\n")
        header = (f"{'qps':>5}  {'tok/s':>6}  {'ttft_p50':>9}  {'ttft_p99':>9}  {'tpot_p50':>8}  "
                  f"{'itl_p99':>8}  {'lat_p50':>9}  {'lat_p99':>9}")
        if has_metrics:
            header += f"  {'wait':>4}  {'run':>4}  {'pre':>4}  {'mem_mb':>7}"
        print(header)

        for qps in [float(x) for x in args.qps_list.split(",")]:
            schedule = workload.schedule(qps, args.num_requests, f"qps{qps:g}")
            stop = asyncio.Event()
            sampler = asyncio.create_task(sample_metrics(client, base, stop)) if has_metrics else None
            results = await run_load(client, base, args.target, schedule, args.max_tokens)
            stop.set()
            server = summarize_server(await sampler) if sampler else None
            s = summarize(results)
            levels.append({"qps": qps, "summary": s, "server": server,
                           "requests": [request_record(r) for r in results]})

            line = (f"{qps:>5g}  {s['throughput']:>6.0f}  {_ms(s['ttft_p50'])}  {_ms(s['ttft_p99'])}  "
                    f"{s['tpot_p50']:>6.1f}ms  {s['itl_p99']:>6.1f}ms  {_ms(s['lat_p50'])}  {_ms(s['lat_p99'])}")
            if server:
                line += (f"  {server['max_waiting']:>4}  {server['max_running']:>4}  "
                         f"{server['preemptions']:>4}  {server['max_active_mb']:>7.0f}")
            print(line, flush=True)

    buckets = list(levels[0]["summary"]["by_bucket"])
    if len(buckets) > 1:
        print(f"\nttft p50 / p99 by bucket\n{'qps':>5}" + "".join(f"  {b:>17}" for b in buckets))
        for lv in levels:
            bb = lv["summary"]["by_bucket"]
            print(f"{lv['qps']:>5g}" + "".join(
                f"  {bb[b]['ttft_p50']:>7.0f} / {bb[b]['ttft_p99']:>5.0f}ms" if b in bb else f"  {'-':>17}"
                for b in buckets))

    sha = (env["git_sha"] or "nogit")[:7] + ("-dirty" if env["git_dirty"] else "")
    out = Path(args.out) if args.out else RESULTS / args.spec / f"{args.target}-{sha}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "out"}
    out.write_text(json.dumps({"env": env, "config": config, "workload": workload.describe(),
                               "levels": levels}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
