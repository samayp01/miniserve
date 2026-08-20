import mlx.core as mx
from mlx_lm import load
from cache import make_prompt_cache, KVCache

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(MODEL_NAME)
WEIGHTS_BYTES = mx.get_active_memory()

def prefill(prompt) -> tuple[mx.array, KVCache]:
    if isinstance(prompt, str):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        prompt = mx.array(ids)

    cache = make_prompt_cache(model)
    logits = model(prompt[None], cache=cache)
    mx.eval(logits)
    
    return logits, cache

def generate_tokens(prompt, max_tokens=128, stop_on_eos=True):
    logits, cache = prefill(prompt)

    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = token.item()

        if stop_on_eos and token_id == tokenizer.eos_token_id:
            break
        
        yield token_id

        logits = model(token[None], cache=cache)
        mx.eval(logits)

def generate(prompt, max_tokens=128, stop_on_eos=True):
    token_ids = list(generate_tokens(prompt, max_tokens=max_tokens, stop_on_eos=stop_on_eos))
    return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
