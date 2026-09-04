class BlockAllocator:
    def __init__(self, num_blocks):
        self.free = list(range(num_blocks))
    
    def allocate(self, num_blocks):
        if len(self.free) < num_blocks:
            raise RuntimeError("Not enough free blocks to allocate")
        allocated = self.free[:num_blocks]
        self.free = self.free[num_blocks:]
        return allocated
    
    def release(self, blocks):
        self.free.extend(blocks)