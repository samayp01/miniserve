import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache as builtin_cache
from src.model_runner import model, tokenizer
from src.cache.cache import make_prompt_cache as my_cache

def _greedy(cache, n=30):
    ids = mx.array(tokenizer.encode("The capital of France is"))
    logits = model(ids[None], cache=cache)
    out = []
    for _ in range(n):
        t = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(t.item())
        logits = model(t[None], cache=cache)
    return out

def test_custom_cache_matches_builtin():
    assert _greedy(my_cache(model)) == _greedy(builtin_cache(model))
