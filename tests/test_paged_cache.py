import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache as builtin_cache
from engine import model, tokenizer
from paged_cache import make_block_pools, make_paged_cache

def _greedy(cache, n=30):
    ids = mx.array(tokenizer.encode("The capital of France is"))
    logits = model(ids[None], cache=cache)
    out = []
    for _ in range(n):
        t = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(t.item())
        logits = model(t[None], cache=cache)
    return out

def test_paged_cache_matches_builtin():
    # block_size=4 forces the ~36-token sequence across ~9 blocks -> exercises real paging
    pools = make_block_pools(model, num_blocks=64, block_size=4)
    paged = make_paged_cache(pools)
    assert _greedy(paged) == _greedy(builtin_cache(model))
