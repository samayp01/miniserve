import time
from collections import deque
from math import ceil
from weakref import WeakValueDictionary

import mlx.core as mx

from src.engine.model_runner import prefill_chunk, batched_decode
from src.cache.paged_cache import make_paged_cache
from src.engine.encode_cache import EncodeCache
from src.engine.scheduler import FifoScheduler

class Engine:
    def __init__(self, pools, adapter, max_batch=32, static=False, chunk_size=512, encode_budget=4, encode_cache_mb=1024, scheduler=None):
        self.adapter = adapter
        self.scheduler = scheduler or FifoScheduler()
        self.encode_budget = encode_budget
        self.encode_cache = EncodeCache(encode_cache_mb * 2**20)
        self.partial = {}
        self.media_data = WeakValueDictionary()
        self.waiting = deque()
        self.running = deque()
        self.pools = pools
        self.max_batch = max_batch
        self.static = static
        self.chunk_size = chunk_size
        self.block_size = pools[0].block_size
        self.capacity = pools[0].num_blocks
        self.preemptions = 0

    def add_request(self, req):
        if req.max_output_tokens < 1:
            raise ValueError(f"max_output_tokens must be at least 1, got {req.max_output_tokens}")
        needed = self._max_blocks_for(req)
        if needed > self.capacity:
            raise ValueError(
                f"request needs up to {needed} blocks ({len(req.prompt_tokens)} prompt "
                f"+ {req.max_output_tokens} output tokens) but the pool holds {self.capacity}"
            )
        for item in req.media:
            if item.key is not None and item.data is not None:
                item.data = self.media_data.setdefault(item.key, item.data)
        self.waiting.append(req)

    def abort(self, req):
        if req in self.waiting:
            self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
        if req.cache is not None:
            for c in req.cache:
                c.release()
            req.cache = None
        self._stash(req)

    def _needed(self, key):
        return any(item.key == key and not item.ready for r in [*self.running, *self.waiting] for item in r.media)

    def _stash(self, req):
        for item in req.media:
            if item.key is None or item.ready:
                continue
            if not self._needed(item.key):
                self.partial.pop(item.key, None)
            elif item.encoded:
                self.partial[item.key] = (item.embeds, item.encoded)

    def _free_blocks(self):
        reserved = sum(self._blocks_for(r) - len(r.cache[0].block_table)
                       for r in self.running if not r.prefilled)
        return len(self.pools[0].allocator.free) - reserved

    def _blocks_for(self, req):
        return ceil((len(req.prompt_tokens) + len(req.output_tokens)) / self.block_size)

    def _max_blocks_for(self, req):
        return ceil((len(req.prompt_tokens) + req.max_output_tokens) / self.block_size)

    def _blocks_needed(self, reqs):
        return sum(1 for r in reqs if r.cache[0].offset % self.block_size == 0)

    def _preempt(self, req):
        self.preemptions += 1
        for c in req.cache:
            c.release()
        req.cache = None
        req.prefilled = False
        req.prefill_pos = 0
        self.running.remove(req)
        self.waiting.appendleft(req)

    def _record(self, req, token):
        if token == self.adapter.eos_token or len(req.output_tokens) >= req.max_output_tokens:
            req.mark_done()
        else:
            req.yield_token(token)

    def _lookup(self, req):
        for item in req.media:
            if item.key is None or item.ready or item.encoded:
                continue
            cached = self.encode_cache.get(item.key)
            if cached is not None:
                self._fill(item, cached)

    def _fill(self, item, embeds):
        item.embeds, item.encoded, item.data, item.hit = embeds, item.pieces, None, True

    def _encode(self):
        budget = self.encode_budget
        claimed = set()
        for req in self.scheduler.order(self.running, self):
            for item in req.media:
                if item.ready:
                    continue
                if item.key is not None:
                    cached = self.encode_cache.peek(item.key)
                    if cached is not None:
                        self._fill(item, cached)
                        continue
                    if item.key in claimed:
                        continue
                    claimed.add(item.key)
                    saved = self.partial.pop(item.key, None)
                    if saved and saved[1] > item.encoded:
                        item.embeds, item.encoded = saved
                if budget == 0:
                    continue
                end = min(item.pieces, item.encoded + budget)
                start = time.perf_counter()
                vectors = self.adapter.encode(item, item.encoded, end)
                mx.eval(vectors)
                self.scheduler.observe("piece", (time.perf_counter() - start) * 1000, end - item.encoded)
                item.embeds = vectors if item.embeds is None else mx.concatenate([item.embeds, vectors])
                budget -= end - item.encoded
                item.encoded = end
                if item.ready:
                    item.data = None
                    if item.key is not None:
                        self.encode_cache.put(item.key, item.embeds)

    def step(self):
        if not (self.static and self.running):
            free = self._free_blocks()
            for req in self.scheduler.order(self.waiting, self):
                if len(self.running) >= self.max_batch or self._blocks_for(req) > free:
                    break
                self.waiting.remove(req)
                req.cache = make_paged_cache(self.pools)
                req.mark_admitted()
                self._lookup(req)
                self.running.append(req)
                free -= self._blocks_for(req)

        self._encode()
        ready = [r for r in self.running if r.media_ready]
        for req in ready:
            req.mark_encoded()
        prefilling = self.scheduler.order([r for r in ready if not r.prefilled], self)
        decoding = [r for r in self.running if r.prefilled and not r.done]

        budget = self.chunk_size
        for req in prefilling:
            if budget <= 0:
                break
            start = time.perf_counter()
            token, used = prefill_chunk(req, budget, self.adapter)
            self.scheduler.observe("token", (time.perf_counter() - start) * 1000, used)
            self._record(req, token)
            budget -= used

        while decoding and self._blocks_needed(decoding) > self._free_blocks():
            self._preempt(decoding.pop())

        if decoding:
            for req, token in zip(decoding, batched_decode(decoding, self.adapter)):
                self._record(req, token)

        finished = [r for r in self.running if r.done]
        self.running = deque(r for r in self.running if not r.done)
        for req in finished:
            for c in req.cache:
                c.release()

    def run(self):
        while self.waiting or self.running:
            self.step()
