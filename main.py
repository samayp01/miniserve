import time
import statistics
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(MODEL_NAME)

def run(user_text="Hi", max_tokens=128, stop_on_eos=False, token_ids=None):
    if token_ids is not None:
        prompt = token_ids
    else:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
        )
        prompt = mx.array(ids)

    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(prompt[None], cache=cache)
    mx.eval(logits)
    prefill_s = time.perf_counter() - t0

    out = []

    t1 = time.perf_counter()
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = token.item() # to synchronize decode

        if stop_on_eos and token_id == tokenizer.eos_token_id:
            break

        out.append(token_id)

        logits = model(token[None], cache=cache)

    mx.eval(logits) # eval last step's logits

    decode_s = time.perf_counter() - t1

    return {
        "prompt_tokens": prompt.size,
        "gen_tokens": len(out),
        "prefill_ms": round(prefill_s * 1000, 1),
        "ms_per_tok": round(decode_s * 1000 / len(out), 2) if out else None,
        # "characters": tokenizer.decode(out),
    }

def make_synthetic_prompt(n):
    tid = tokenizer.encode("the")[-1]
    return mx.array([tid] * n)

def measure(token_ids, max_tokens=128, reps=5):
    # warm up run
    run(max_tokens=max_tokens, token_ids=token_ids)

    prefill, per_tok = [], []
    for _ in range(reps):
        r = run(max_tokens=max_tokens, token_ids=token_ids)
        prefill.append(r["prefill_ms"])
        per_tok.append(r["ms_per_tok"])

    # median = middle value, so a single unlucky run doesn't skew the number
    return {
        "prompt_tokens": token_ids.size,
        "prefill_ms": round(statistics.median(prefill), 1),
        "ms_per_tok": round(statistics.median(per_tok), 2),
    }

if __name__ == "__main__":
    lengths = [64, 256, 1024, 4096]
    rows = [measure(make_synthetic_prompt(n)) for n in lengths]

    # the Rung 0 artifact: prefill grows with prompt length, decode stays flat
    print(f"{'prompt_tok':>10} {'prefill_ms':>11} {'ms/tok':>8} {'prefill/tok':>12}")
    for r in rows:
        prefill_per_tok = r["prefill_ms"] / r["prompt_tokens"]
        print(
            f"{r['prompt_tokens']:>10} {r['prefill_ms']:>11.1f} "
            f"{r['ms_per_tok']:>8.2f} {prefill_per_tok:>12.3f}"
        )