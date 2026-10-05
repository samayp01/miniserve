import mlx.core as mx
import pytest

from src.cache.paged_cache import make_block_pools, make_paged_cache
from src.cache.swap import NotEnoughBlocks, SwapStore
from tests.unit.fakes import FakeAdapter


def _filled(pools, tokens, dtype):
    caches = make_paged_cache(pools)
    for c in caches:
        kv = mx.random.normal((1, 2, tokens, 4)).astype(dtype)
        c.store(kv, kv * 2)
    mx.eval([c.pool.k_pool for c in caches])
    return caches


def _contents(caches, dtype):
    empty = mx.zeros((1, 2, 0, 4), dtype)
    return [c.update_and_fetch(empty, empty) for c in caches]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_swap_round_trip_is_exact(dtype):
    pools = make_block_pools(FakeAdapter(num_layers=3), 32, 16)
    caches = _filled(pools, 40, dtype)
    before = _contents(caches, dtype)
    store = SwapStore()
    store.save("r1", caches)
    assert all(len(p.allocator.free) == 32 for p in pools)
    store.load("r1", caches)
    after = _contents(caches, dtype)
    assert all(c.offset == 40 for c in caches)
    for (k0, v0), (k1, v1) in zip(before, after):
        assert mx.array_equal(k0, k1) and mx.array_equal(v0, v1)
    assert store.bytes == 0 and not any(store.directory.iterdir())


def test_discard_removes_a_swapped_request():
    pools = make_block_pools(FakeAdapter(), 32, 16)
    store = SwapStore()
    store.save("r1", _filled(pools, 20, mx.float16))
    store.discard("r1")
    assert store.bytes == 0 and store.meta == {} and not any(store.directory.iterdir())


def test_load_without_enough_blocks_changes_nothing():
    pools = make_block_pools(FakeAdapter(num_layers=3), 4, 16)
    store = SwapStore()
    store.save("r1", _filled(pools, 40, mx.float16))
    blocker = _filled(pools, 40, mx.float16)
    with pytest.raises(NotEnoughBlocks):
        store.load("r1", make_paged_cache(pools))
    assert all(len(p.allocator.free) == 1 for p in pools)
    assert "r1" in store.meta and any(store.directory.iterdir())
    for c in blocker:
        c.release()
    store.load("r1", make_paged_cache(pools))
    assert all(len(p.allocator.free) == 1 for p in pools)
    assert store.meta == {} and store.bytes == 0
