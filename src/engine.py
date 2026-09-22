from collections import deque
from math import ceil
from src.model_runner import EOS_TOKEN, prefill_chunk, batched_decode
from src.cache.paged_cache import make_paged_cache

class Engine:
    def __init__(self, pools, max_batch=32, static=False, chunk_size=512):
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
        needed = self._max_blocks_for(req)
        if needed > self.capacity:
            raise ValueError(
                f"request needs up to {needed} blocks ({len(req.prompt_tokens)} prompt "
                f"+ {req.max_output_tokens} output tokens) but the pool holds {self.capacity}"
            )
        self.waiting.append(req)

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
        if token == EOS_TOKEN or len(req.output_tokens) >= req.max_output_tokens:
            req.mark_done()
        else:
            req.yield_token(token)

    def step(self):
        if not (self.static and self.running):
            free = self._free_blocks()
            while self.waiting and len(self.running) < self.max_batch and self._blocks_for(self.waiting[0]) <= free:
                req = self.waiting.popleft()
                req.cache = make_paged_cache(self.pools)
                self.running.append(req)
                free -= self._blocks_for(req)

        prefilling = [r for r in self.running if not r.prefilled]
        decoding = [r for r in self.running if r.prefilled and not r.done]

        budget = self.chunk_size
        for req in prefilling:
            if budget <= 0:
                break
            token, used = prefill_chunk(req, budget)
            self._record(req, token)
            budget -= used

        while decoding and self._blocks_needed(decoding) > self._free_blocks():
            self._preempt(decoding.pop())

        if decoding:
            for req, token in zip(decoding, batched_decode(decoding)):
                self._record(req, token)

        finished = [r for r in self.running if r.done]
        self.running = deque(r for r in self.running if not r.done)
        for req in finished:
            for c in req.cache:
                c.release()

    def run(self):
        while self.waiting or self.running:
            self.step()
