import time
import itertools
import mlx.core as mx
from src.model_runner import model, tokenizer
from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools, make_paged_cache

HEADS, DIM = 8, 64  # Llama-3.2-1B kv heads / head dim

def _toks(p):
    return tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)

def max_concurrent_paged(mix, num_blocks, block_size):
    pools = make_block_pools(model, num_blocks, block_size)
    resident = []
    for length in itertools.cycle(mix):
        cache = make_paged_cache(pools)
        kv = mx.zeros((1, HEADS, length, DIM), mx.float16)
        try:
            for c in cache:
                c.update_and_fetch(kv, kv)
        except RuntimeError:
            break
        mx.eval([p.k_pool for p in pools])
        resident.append(cache)
    return len(resident)

def print_paged_concurrency(mix=(8, 16, 32, 128), num_blocks=128, block_size=16):
    mix = list(mix)
    budget = num_blocks * block_size
    paged = max_concurrent_paged(mix, num_blocks, block_size)
    contiguous = budget // max(mix)
    print(f"\npool: {num_blocks} blocks x {block_size} = {budget} tokens/layer,  mix={mix}")
    print(f"{'':>12}{'max concurrent':>16}")
    print(f"{'paged':>12}{paged:>16}")
    print(f"{'contiguous':>12}{contiguous:>16}   (each reserves max_len={max(mix)})")


def _bench(mode, max_batch=4):
    SHORT = ("Say hi in one word.", 8)
    LONG = ("Write a detailed multi-paragraph essay about the Roman empire.", 96)
    pools = make_block_pools(model, num_blocks=4096, block_size=16)
    engine = Engine(pools, max_batch=max_batch, static=(mode == "static"))
    reqs = [Request(_toks((LONG if i % 4 == 0 else SHORT)[0]),
                    max_output_tokens=(LONG if i % 4 == 0 else SHORT)[1]) for i in range(16)]
    for r in reqs:
        engine.add_request(r)

    t0 = time.time()
    engine.run()
    wall = time.time() - t0

    ttfts = sorted(r.first_token_time - r.arrival_time for r in reqs)
    total = sum(len(r.output_tokens) for r in reqs)
    pct = lambda q: ttfts[min(len(ttfts) - 1, int(q * len(ttfts)))]
    return wall, total / wall, pct(0.50), pct(0.99)

def print_scheduling_benchmark():
    print(f"\n{'mode':>11}  {'wall':>6}  {'tok/s':>8}  {'ttft_p50':>9}  {'ttft_p99':>9}")
    for mode in ("static", "continuous"):
        wall, tput, p50, p99 = _bench(mode)
        print(f"{mode:>11}  {wall:5.2f}s  {tput:8.1f}  {p50:8.2f}s  {p99:8.2f}s")


if __name__ == "__main__":
    print_paged_concurrency()
    print_scheduling_benchmark()
