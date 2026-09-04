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


class PagedKVCache:
    def __init__(self, pool):
        self.pool = pool
        self.block_table = []
        self.offset = 0

    def update_and_fetch(self, keys, values):
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

        ids = mx.array(self.block_table)
        k = pool.k_pool[ids].transpose(1, 0, 2, 3).reshape(1, heads, -1, dim)[:, :, :self.offset, :]
        v = pool.v_pool[ids].transpose(1, 0, 2, 3).reshape(1, heads, -1, dim)[:, :, :self.offset, :]
        return k, v

    def release(self):
        self.pool.allocator.release(self.block_table)
        self.block_table = []


def make_block_pools(model, num_blocks, block_size):
    return [BlockPool(num_blocks, block_size) for _ in range(len(model.layers))]

def make_paged_cache(pools):
    return [PagedKVCache(pool) for pool in pools]
