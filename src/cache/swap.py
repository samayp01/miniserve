import fcntl
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np


class NotEnoughBlocks(RuntimeError):
    pass


def _nocache(f):
    if hasattr(fcntl, "F_NOCACHE"):
        fcntl.fcntl(f.fileno(), fcntl.F_NOCACHE, 1)


class SwapStore:
    def __init__(self, directory=None):
        self._tmp = None if directory else tempfile.TemporaryDirectory(prefix="miniserve-swap-")
        self.directory = Path(directory or self._tmp.name)
        self.meta = {}
        self.bytes = 0

    def _path(self, key):
        return self.directory / f"{key}.kv"

    def save(self, key, caches):
        k = mx.stack([c.pool.k_pool[mx.array(c.block_table)] for c in caches])
        v = mx.stack([c.pool.v_pool[mx.array(c.block_table)] for c in caches])
        kv = mx.stack([k, v])
        raw = np.array(kv.view(mx.uint8))
        with open(self._path(key), "wb", buffering=0) as f:
            _nocache(f)
            f.write(memoryview(raw).cast("B"))
        self.meta[key] = (kv.dtype, caches[0].offset, len(caches[0].block_table), raw.nbytes, raw.shape)
        self.bytes += raw.nbytes
        for c in caches:
            c.release()
            c.offset = 0
        return raw.nbytes

    def load(self, key, caches):
        dtype, offset, blocks, nbytes, shape = self.meta[key]
        if any(len(c.pool.allocator.free) < blocks for c in caches):
            raise NotEnoughBlocks(f"not enough free blocks to load {key}: need {blocks} per layer")
        raw = np.empty(shape, dtype=np.uint8)
        view = memoryview(raw).cast("B")
        with open(self._path(key), "rb", buffering=0) as f:
            _nocache(f)
            read = 0
            while read < nbytes:
                read += f.readinto(view[read:])
        kv = mx.array(raw).view(dtype)
        for layer, c in enumerate(caches):
            c.block_table = c.pool.allocator.allocate(blocks)
            ids = mx.array(c.block_table)
            c.pool.k_pool[ids] = kv[0, layer]
            c.pool.v_pool[ids] = kv[1, layer]
            c.offset = offset
        mx.eval([c.pool.k_pool for c in caches] + [c.pool.v_pool for c in caches])
        self.discard(key)

    def discard(self, key):
        entry = self.meta.pop(key, None)
        self._path(key).unlink(missing_ok=True)
        if entry:
            self.bytes -= entry[3]
