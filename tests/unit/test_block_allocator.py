import pytest
from src.cache.block_allocator import BlockAllocator

def test_starts_with_all_blocks_free():
    alloc = BlockAllocator(8)
    assert alloc.free == list(range(8))

def test_allocate_shrinks_free_and_returns_distinct_blocks():
    alloc = BlockAllocator(8)
    blocks = alloc.allocate(3)
    assert blocks == [0, 1, 2]
    assert alloc.free == [3, 4, 5, 6, 7]

def test_allocate_raises_when_not_enough_blocks():
    alloc = BlockAllocator(2)
    with pytest.raises(RuntimeError):
        alloc.allocate(3)

def test_release_returns_blocks_to_free():
    alloc = BlockAllocator(4)
    blocks = alloc.allocate(4)
    assert alloc.free == []
    alloc.release(blocks)
    assert sorted(alloc.free) == [0, 1, 2, 3]

def test_released_blocks_can_be_reallocated():
    alloc = BlockAllocator(4)
    alloc.allocate(4)
    alloc.release([1, 2])
    assert alloc.allocate(2) == [1, 2]
