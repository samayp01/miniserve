import statistics
import mlx.core as mx
from engine import generate, tokenizer

def make_synthetic_prompt(n):
    tid = tokenizer.encode("the")[-1]
    return mx.array([tid] * n)

def benchmark(token_ids, max_tokens=128, trials=5):
    generate(token_ids, max_tokens=max_tokens, stop_on_eos=False) # warmup run
    prefill, per_tok = [], []

    for _ in range(trials):
        _, s = generate(token_ids, max_tokens=max_tokens, stop_on_eos=False)
        prefill.append(s["prefill_ms"])
        per_tok.append(s["ms_per_tok"])

    return {
        "prompt_tokens": token_ids.size,
        "gen_tokens": max_tokens,
        "prefill_ms": round(statistics.mean(prefill), 1),
        "ms_per_tok": round(statistics.mean(per_tok), 2),
    }

if __name__ == "__main__":
    rows = [benchmark(make_synthetic_prompt(n)) for n in (64, 256, 1024, 4096)]
    print(f"{'prompt_tok':>12} {'prefill_ms':>12} {'ms/tok':>10} {'prefill/tok':>14}")
    for r in rows:
        ppt = r["prefill_ms"] / r["prompt_tokens"]
        print(f"{r['prompt_tokens']:>12} {r['prefill_ms']:>12.2f} {r['ms_per_tok']:>10.2f} {ppt:>14f}")
