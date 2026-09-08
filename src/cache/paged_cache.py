from math import ceil
import mlx.core as mx
from src.cache.block_allocator import BlockAllocator

class BlockPool:
    def __init__(self, num_blocks, block_size):
        self.allocator = BlockAllocator(num_blocks)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.k_pool = None
        self.v_pool = None

    def ensure(self, heads, dim, dtype):
        if self.k_pool is None:
            shape = (self.num_blocks, heads, self.block_size, dim)
            self.k_pool = mx.zeros(shape, dtype=dtype)
            self.v_pool = mx.zeros(shape, dtype=dtype)

    def has_free_blocks(self):
        return len(self.allocator.free) > 0


class PagedKVCache:
    def __init__(self, pool):
        self.pool = pool
        self.block_table = []
        self.offset = 0

    def store(self, keys, values):
        pool = self.pool
        _, heads, L, dim = keys.shape
        pool.ensure(heads, dim, keys.dtype)

        blocks_needed = ceil((self.offset + L) / pool.block_size)
        if blocks_needed > len(self.block_table):
            self.block_table += pool.allocator.allocate(blocks_needed - len(self.block_table))

        for i in range(L):
            p = self.offset + i
            b, s = self.block_table[p // pool.block_size], p % pool.block_size
            pool.k_pool[b, :, s, :] = keys[0, :, i, :]
            pool.v_pool[b, :, s, :] = values[0, :, i, :]
        self.offset += L

    def update_and_fetch(self, keys, values):
        self.store(keys, values)
        pool, heads, dim = self.pool, keys.shape[1], keys.shape[3]
        ids = mx.array(self.block_table)
        k = pool.k_pool[ids].transpose(1, 0, 2, 3).reshape(1, heads, -1, dim)[:, :, :self.offset, :]
        v = pool.v_pool[ids].transpose(1, 0, 2, 3).reshape(1, heads, -1, dim)[:, :, :self.offset, :]
        return k, v

    def release(self):
        self.pool.allocator.release(self.block_table)
        self.block_table = []


class BatchedPagedCache:
    def __init__(self, caches):
        self.caches = caches

    @property
    def offset(self):
        return mx.array([c.offset for c in self.caches])

    def make_mask(self, N, return_array=False, window_size=None):
        bs = self.caches[0].pool.block_size
        lengths = [c.offset + N for c in self.caches]
        max_nb = max(ceil(l / bs) for l in lengths)
        T = max_nb * bs
        real = mx.array(lengths)[:, None]
        allowed = mx.arange(T)[None, :] < real
        return allowed[:, None, None, :]

    def update_and_fetch(self, keys, values):
        _, heads, _, dim = keys.shape
        for i, c in enumerate(self.caches):
            c.store(keys[i:i+1], values[i:i+1])

        pool = self.caches[0].pool
        bs, N = pool.block_size, len(self.caches)
        max_nb = max(len(c.block_table) for c in self.caches)
        grid = mx.array([c.block_table + [0] * (max_nb - len(c.block_table)) for c in self.caches])
        k = pool.k_pool[grid].transpose(0, 2, 1, 3, 4).reshape(N, heads, max_nb * bs, dim)
        v = pool.v_pool[grid].transpose(0, 2, 1, 3, 4).reshape(N, heads, max_nb * bs, dim)
        return k, v


def make_block_pools(model, num_blocks, block_size):
    return [BlockPool(num_blocks, block_size) for _ in range(len(model.layers))]

def make_paged_cache(pools):
    return [PagedKVCache(pool) for pool in pools]
