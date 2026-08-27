import time
import statistics
import mlx.core as mx
from engine import _prefill_ids, _decode, tokenizer, WEIGHTS_BYTES

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
    print_memory_experiment()
