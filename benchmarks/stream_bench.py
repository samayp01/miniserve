import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

RESULTS = Path(__file__).parent / "results" / "stream"
WATCH = ("active_mb", "peak_rss_mb", "encode_cache_mb", "prefetch_backlog", "waiting", "running")


def render(stream, index, size=512):
    image = Image.new("RGB", (size, size), ((40 * stream) % 200 + 20, 60, 160))
    x = (index * 7) % (size - 120)
    draw = ImageDraw.Draw(image)
    draw.ellipse((x, 180, x + 120, 300), fill=(220, 30, 30))
    draw.text((10, 10), f"{stream}-{index}", fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def camera(client, stream_id, stream, args, stop):
    index, start = 0, time.perf_counter()
    while not stop.is_set():
        data = await asyncio.to_thread(render, stream, index)
        await client.post(f"/streams/{stream_id}/frames", json={"data": data})
        index += 1
        await asyncio.sleep(max(0.0, start + index / args.camera_fps - time.perf_counter()))


async def asker(client, stream_id, args, stop, asks, t0):
    await asyncio.sleep(args.ask_every)
    while not stop.is_set():
        sent, first, done = time.perf_counter(), None, {}
        body = {"prompt": "What is happening in these frames?", "max_tokens": args.max_tokens}
        async with client.stream("POST", f"/streams/{stream_id}/ask", json=body) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if "delta" in event and first is None:
                        first = time.perf_counter()
                    if event.get("done"):
                        done = event
        asks.append({"t": round(sent - t0, 2), "status": response.status_code,
                     "ttft_ms": round((first - sent) * 1000, 1) if first else None,
                     "latency_ms": round((time.perf_counter() - sent) * 1000, 1),
                     "media": done.get("media"), "media_hits": done.get("media_hits")})
        await asyncio.sleep(args.ask_every)


async def sampler(client, stop, samples, t0, warmup):
    while not stop.is_set():
        snapshot = (await client.get("/metrics")).json()
        snapshot["t"] = round(time.perf_counter() - t0, 2)
        if snapshot["t"] >= warmup:
            samples.append(snapshot)
        await asyncio.sleep(0.5)


def thirds(samples, key):
    third = max(1, len(samples) // 3)
    mean = lambda rows: sum(r[key] for r in rows) / len(rows)
    return mean(samples[:third]), mean(samples[-third:]), max(r[key] for r in samples)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--streams", type=int, default=3)
    ap.add_argument("--camera-fps", type=float, default=10)
    ap.add_argument("--fps", type=float, default=1)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--ask-every", type=float, default=5)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=float, default=10)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    samples, asks, stop = [], [], asyncio.Event()
    async with httpx.AsyncClient(base_url=args.url, timeout=None) as client:
        ids = [(await client.post("/streams", json={"fps": args.fps, "window": args.window})).json()["stream_id"]
               for _ in range(args.streams)]
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(sampler(client, stop, samples, t0, args.warmup))]
        tasks += [asyncio.create_task(camera(client, sid, i, args, stop)) for i, sid in enumerate(ids)]
        tasks += [asyncio.create_task(asker(client, sid, args, stop, asks, t0)) for sid in ids]
        await asyncio.sleep(args.warmup + args.duration)
        stop.set()
        await asyncio.gather(*tasks)
        closed = [(await client.delete(f"/streams/{sid}")).json() for sid in ids]
    asks = [a for a in asks if a["t"] >= args.warmup]

    print(f"\n{args.streams} streams, camera {args.camera_fps:g} fps, sampled {args.fps:g} fps, "
          f"window {args.window}, {args.duration:g}s after {args.warmup:g}s warm-up\n")
    print(f"{'metric':>18} {'first third':>12} {'last third':>11} {'max':>8}")
    for key in WATCH:
        first, last, peak = thirds(samples, key)
        print(f"{key:>18} {first:>12.1f} {last:>11.1f} {peak:>8.1f}")
    frames = {k: sum(c[k] for c in closed) for k in ("kept", "skipped", "dropped")}
    ttfts = sorted(a["ttft_ms"] for a in asks if a["ttft_ms"] is not None)
    hit = sum(a["media_hits"] or 0 for a in asks) / max(1, sum(a["media"] or 0 for a in asks))
    print(f"\nframes {frames}")
    print(f"asks {len(asks)}: ttft p50 {ttfts[len(ttfts) // 2]:.0f}ms, max {ttfts[-1]:.0f}ms, "
          f"window frames already encoded {hit:.0%}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.tag}.json"
    out.write_text(json.dumps({"config": vars(args), "frames": frames, "asks": asks, "samples": samples}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
