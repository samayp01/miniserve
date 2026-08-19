import time
import mlx.core as mx
from mlx_lm import load
# from mlx_lm.models.cache import make_prompt_cache
from cache import make_prompt_cache

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(MODEL_NAME)

def generate(prompt, max_tokens=128, stop_on_eos=True):
    if isinstance(prompt, str):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
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
    first_token_s = None

    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = token.item()

        if stop_on_eos and token_id == tokenizer.eos_token_id:
            break

        if first_token_s is None:
            first_token_s = time.perf_counter() - t0

        out.append(token_id)
        logits = model(token[None], cache=cache)

    mx.eval(logits) # eval last step's logits
    decode_s = time.perf_counter() - t1
    text = tokenizer.decode(out, clean_up_tokenization_spaces=False)
    stats = {
        "prompt_tokens": prompt.size,
        "gen_tokens": len(out),
        "prefill_ms": round(prefill_s * 1000, 1),
        "ttft_ms": round(first_token_s * 1000, 1) if first_token_s is not None else None,
        "ms_per_tok": round(decode_s * 1000 / len(out), 2) if out else None,
    }

    return text, stats
