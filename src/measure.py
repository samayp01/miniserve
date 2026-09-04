import time
import itertools
import statistics
import mlx.core as mx
from src.model_runner import model, _prefill_ids, _decode, tokenizer, WEIGHTS_BYTES
from src.cache.paged_cache import make_block_pools, make_paged_cache

HEADS, DIM = 8, 64   # Llama-3.2-1B kv heads / head dim

def make_synthetic_prompt(n):
    tid = tokenizer.encode("the")[-1]
    return [tid] * n

def _time_prefill(token_ids):
    t = time.perf_counter()
    _prefill_ids([token_ids])
    return time.perf_counter() - t

def _time_ttft_and_decode(ids, max_tokens):
    t = time.perf_counter()
    logits, cache = _prefill_ids([ids])
    gen = _decode(logits, cache, max_tokens, stop_on_eos=False)
    next(gen)
    ttft_s = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(max_tokens - 1):
        next(gen)
    return ttft_s, time.perf_counter() - t

def memory_usage(ids, max_tokens=2048, sample_every=256):
    logits, cache = _prefill_ids([ids])
    gen = _decode(logits, cache, max_tokens, stop_on_eos=False)
    samples = []
    mx.reset_peak_memory()
    for i, _ in enumerate(gen, start=1):
        if i % sample_every == 0:
            active = mx.get_active_memory()
            gap = mx.get_peak_memory() - active
            samples.append((i, active, gap))
            mx.reset_peak_memory()
    return samples

def memory_report(samples, weights_bytes):
    xs = [t for t, *_ in samples]
    ys = [active for _, active, *_ in samples]
    bytes_per_tok = statistics.linear_regression(xs, ys).slope

    predicted = 16 * 8 * 64 * 2 * 2
    working_set = mx.device_info()["max_recommended_working_set_size"]
    free = working_set - weights_bytes
    ceiling_tokens = free / bytes_per_tok

    return {
        "bytes_per_tok": round(bytes_per_tok),
        "predicted_bytes_per_tok": predicted,
        "ceiling_tokens": int(ceiling_tokens),
        "working_set_gb": round(working_set / 1e9, 1),
    }

def benchmark(token_ids, max_tokens=128, trials=5):
    logits, cache = _prefill_ids([token_ids])
    list(_decode(logits, cache, max_tokens=max_tokens, stop_on_eos=False))

    prefill_s, ttft_s, decode_s = [], [], []
    for _ in range(trials):
        prefill_s.append(_time_prefill(token_ids))
        tt, dec = _time_ttft_and_decode(token_ids, max_tokens)
        ttft_s.append(tt)
        decode_s.append(dec)

    prefill_med = statistics.median(prefill_s)
    ttft_med = statistics.median(ttft_s)
    decode_med = statistics.median(decode_s)
    ms_per_tok = decode_med * 1000 / (max_tokens - 1)

    return {
        "prompt_tokens": len(token_ids),
        "gen_tokens": max_tokens,
        "prefill_ms": round(prefill_med * 1000, 1),
        "ttft_ms": round(ttft_med * 1000, 1),
        "ms_per_tok": round(ms_per_tok, 2),
        "decode_tok_per_sec": round(1000 / ms_per_tok, 1),
        "prefill_tok_per_sec": round(len(token_ids) / prefill_med, 1),
    }

def print_timing_sweep(sizes=(64, 256, 1024, 4096)):
    rows = [benchmark(make_synthetic_prompt(n)) for n in sizes]
    print(
        f"{'prompt_tok':>10} {'prefill_ms':>11} {'ttft_ms':>9} {'ms/tok':>8} "
        f"{'decode_tok/s':>13} {'prefill_tok/s':>14}"
    )
    for r in rows:
        print(
            f"{r['prompt_tokens']:>10} {r['prefill_ms']:>11.1f} {r['ttft_ms']:>9.1f} "
            f"{r['ms_per_tok']:>8.2f} {r['decode_tok_per_sec']:>13.1f} "
            f"{r['prefill_tok_per_sec']:>14.1f}"
        )

def bench_batch(batch_size, prompt_len=64, max_tokens=64, trials=3):
    batch = [make_synthetic_prompt(prompt_len)] * batch_size

    logits, cache = _prefill_ids(batch)
    list(_decode(logits, cache, max_tokens, stop_on_eos=False))

    times = []
    for _ in range(trials):
        logits, cache = _prefill_ids(batch)
        gen = _decode(logits, cache, max_tokens, stop_on_eos=False)
        next(gen)
        t = time.perf_counter()
        for _ in range(max_tokens - 1):
            next(gen)
        times.append(time.perf_counter() - t)

    decode_s = statistics.median(times)
    per_seq_tokens = max_tokens - 1
    return {
        "batch": batch_size,
        "agg_tok_per_sec": round(batch_size * per_seq_tokens / decode_s, 1),
        "ms_per_step": round(decode_s * 1000 / per_seq_tokens, 2),
    }

def print_batch_sweep(sizes=(1, 2, 4, 8, 16)):
    rows = [bench_batch(b) for b in sizes]
    print(f"\n{'batch':>6} {'agg_tok/s':>11} {'ms/step':>9}")
    for r in rows:
        print(f"{r['batch']:>6} {r['agg_tok_per_sec']:>11.1f} {r['ms_per_step']:>9.2f}")

def padding_waste(lengths, trials=3):
    batch = [make_synthetic_prompt(n) for n in lengths]
    real = sum(lengths)
    total = len(lengths) * max(lengths)

    _prefill_ids(batch)
    times = []
    for _ in range(trials):
        t = time.perf_counter()
        _prefill_ids(batch)
        times.append(time.perf_counter() - t)
    prefill_s = statistics.median(times)

    return {
        "lengths": lengths,
        "real_tok": real,
        "pad_tok": total - real,
        "waste_pct": (total - real) / total * 100,
        "prefill_ms": prefill_s * 1000,
    }

def print_padding_waste():
    workloads = [
        [64, 64, 64, 64],
        [16, 48, 80, 112],
        [8, 8, 8, 488],
    ]
    print(f"\n{'lengths':>22} {'real':>6} {'pad':>6} {'waste%':>7} {'prefill_ms':>11} {'wasted_ms':>10}")
    for w in workloads:
        r = padding_waste(w)
        wasted_ms = r["prefill_ms"] * r["waste_pct"] / 100
        print(
            f"{str(r['lengths']):>22} {r['real_tok']:>6} {r['pad_tok']:>6} "
            f"{r['waste_pct']:>7.1f} {r['prefill_ms']:>11.1f} {wasted_ms:>10.1f}"
        )

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
    print(f"-> paged holds {paged / contiguous:.1f}x more sequences in the same memory")

def print_memory_experiment():
    samples = memory_usage(make_synthetic_prompt(8))
    report = memory_report(samples, WEIGHTS_BYTES)

    print(f"\n{'tokens':>8} {'active_MB':>11} {'concat_gap_MB':>14}")
    for n, active, gap in samples:
        print(f"{n:>8} {active / 1e6:>11.1f} {gap / 1e6:>14.1f}")

    print(
        f"\nbytes/token: {report['bytes_per_tok']:,} "
        f"(predicted {report['predicted_bytes_per_tok']:,})"
    )
    print(
        f"ceiling: {report['ceiling_tokens']:,} tokens "
        f"in {report['working_set_gb']}GB working set"
    )

if __name__ == "__main__":
    print_timing_sweep()
    print_batch_sweep()
    print_padding_waste()
    print_paged_concurrency()
    print_memory_experiment()
