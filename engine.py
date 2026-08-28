import mlx.core as mx
from mlx_lm import load
from cache import make_prompt_cache, KVCache

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(MODEL_NAME)
WEIGHTS_BYTES = mx.get_active_memory()

def pad_batch(batch: list[list[int]]) -> tuple[mx.array, mx.array]:
    max_len = max(len(ids) for ids in batch)
    padded_batch, pad_lens = [], []
    for ids in batch:
        pad = max_len - len(ids)
        padded_batch.append(([0] * pad) + ids)
        pad_lens.append(pad)
    return mx.array(padded_batch), mx.array(pad_lens)

def _prefill_ids(batch: list[list[int]]) -> tuple[mx.array, list[KVCache]]:
    prompt_grid, pad_lens = pad_batch(batch)
    cache = make_prompt_cache(model)
    if int(pad_lens.max()) > 0:
        for c in cache:
            c.pad_lens = pad_lens
    logits = model(prompt_grid, cache=cache)
    mx.eval(logits)

    return logits, cache

def prefill(prompts: list[str]) -> tuple[mx.array, list[KVCache]]:
    assert all(isinstance(p, str) for p in prompts), "prefill expects list[str]"
    batch = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    return _prefill_ids(batch)

def _decode(logits, cache, max_tokens, stop_on_eos):
    N = logits.shape[0]
    done = [False] * N

    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        ids = token.tolist()

        step = []
        for i, token_id in enumerate(ids):
            if done[i]:
                step.append(None)
            elif stop_on_eos and token_id == tokenizer.eos_token_id:
                done[i] = True
                step.append(None)
            else:
                step.append(token_id)
        
        yield step
        if all(done):
            break

        logits = model(token[:, None], cache=cache)
        mx.eval(logits)

def generate_tokens(prompts, max_tokens=128, stop_on_eos=True):
    logits, cache = prefill(prompts)
    yield from _decode(logits, cache, max_tokens, stop_on_eos)

def generate(prompts, max_tokens=128, stop_on_eos=True):
    N = len(prompts)
    outputs = [[] for _ in range(N)]

    for step in generate_tokens(prompts, max_tokens=max_tokens, stop_on_eos=stop_on_eos):
        for i, token_id in enumerate(step):
            if token_id is not None:
                outputs[i].append(token_id)
    
    return [tokenizer.decode(ids, clean_up_tokenization_spaces=False) for ids in outputs]
