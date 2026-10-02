import mlx.core as mx

from src.engine.encode_cache import EncodeCache


def _embeds(rows):
    return mx.zeros((rows, 1), dtype=mx.float32)


def test_hit_returns_the_stored_vectors_and_counts_it():
    cache = EncodeCache(1024)
    embeds = _embeds(4)
    cache.put("a", embeds)
    assert cache.get("a") is embeds
    assert cache.get("b") is None
    assert (cache.hits, cache.misses) == (1, 1)


def test_evicts_least_recently_used_when_over_budget():
    cache = EncodeCache(3 * 16)
    for key in "abc":
        cache.put(key, _embeds(4))
    cache.get("a")
    cache.put("d", _embeds(4))
    assert list(cache.entries) == ["c", "a", "d"]
    assert cache.bytes == 3 * 16


def test_zero_budget_stores_nothing():
    cache = EncodeCache(0)
    cache.put("a", _embeds(4))
    assert cache.get("a") is None
    assert cache.bytes == 0
