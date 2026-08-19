import time
import statistics
import mlx.core as mx
from engine import prefill, generate_tokens, tokenizer

def make_synthetic_prompt(n):
    tid = tokenizer.encode("the")[-1]
    return mx.array([tid] * n)

def _time_prefill(token_ids):
    t = time.perf_counter()
    prefill(token_ids)
    return time.perf_counter() - t

def _time_ttft_and_decode(token_ids, max_tokens):
    gen = generate_tokens(token_ids, max_tokens=max_tokens, stop_on_eos=False)

    t = time.perf_counter()
    next(gen)
    ttft_s = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(max_tokens - 1):
        next(gen)
    decode_s = time.perf_counter() - t

    return ttft_s, decode_s

def benchmark(token_ids, max_tokens=128, trials=5):
    list(generate_tokens(token_ids, max_tokens=max_tokens, stop_on_eos=False))

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
        "prompt_tokens": int(token_ids.size),
        "gen_tokens": max_tokens,
        "prefill_ms": round(prefill_med * 1000, 1),
        "ttft_ms": round(ttft_med * 1000, 1),
        "ms_per_tok": round(ms_per_tok, 2),
        "decode_tok_per_sec": round(1000 / ms_per_tok, 1),
        "prefill_tok_per_sec": round(token_ids.size / prefill_med, 1),
    }

if __name__ == "__main__":
    rows = [benchmark(make_synthetic_prompt(n)) for n in (64, 256, 1024, 4096)]
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
