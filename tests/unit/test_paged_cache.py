import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache as builtin_cache
from src.model_runner import model, tokenizer
from src.cache.paged_cache import (
    BatchedPagedCache,
    PagedKVCache,
    make_block_pools,
    make_paged_cache,
)

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
    # block_size=4 forces the ~36-token sequence across ~9 blocks
    pools = make_block_pools(model, num_blocks=64, block_size=4)
    paged = make_paged_cache(pools)
    assert _greedy(paged) == _greedy(builtin_cache(model))


def _batched(offsets, block_size=16):
    pool = make_block_pools(model, num_blocks=64, block_size=block_size)[0]
    caches = []
    for offset in offsets:
        cache = PagedKVCache(pool)
        cache.offset = offset
        caches.append(cache)
    return BatchedPagedCache(caches)


def test_make_mask_covers_each_sequence_up_to_its_own_length():
    mask = _batched([3, 20]).make_mask(1)
    assert mask.shape == (2, 1, 1, 32)
    assert mask.sum(axis=-1).flatten().tolist() == [4, 21]


def test_make_mask_rejects_multi_token_query():
    with pytest.raises(NotImplementedError, match="causal within the chunk"):
        _batched([3, 20]).make_mask(2)


def test_make_mask_rejects_sliding_window():
    with pytest.raises(NotImplementedError, match="sliding-window"):
        _batched([3, 20]).make_mask(1, window_size=512)
