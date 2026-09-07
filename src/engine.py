from collections import deque
from src.model_runner import EOS_TOKEN, prefill_request, batched_decode
from src.cache.paged_cache import make_paged_cache

class Engine:
    def __init__(self, pools, max_batch=16, static=False):
        self.waiting = deque()
        self.running = deque()
        self.pools = pools
        self.max_batch = max_batch
        self.static = static

    def add_request(self, req):
        self.waiting.append(req)

    def _record(self, req, token):
        if token == EOS_TOKEN or len(req.output_tokens) >= req.max_output_tokens:
            req.mark_done()
        else:
            req.yield_token(token)

    def step(self):
        if not (self.static and self.running):
            while self.waiting and len(self.running) < self.max_batch and all(p.has_free_blocks() for p in self.pools):
                req = self.waiting.popleft()
                req.cache = make_paged_cache(self.pools)
                self.running.append(req)

        new = [r for r in self.running if not r.prefilled]
        ongoing = [r for r in self.running if r.prefilled and not r.done]

        for req in new:
            self._record(req, prefill_request(req))

        if ongoing:
            for req, token in zip(ongoing, batched_decode(ongoing)):
                self._record(req, token)

        finished = [r for r in self.running if r.done]
        self.running = deque(r for r in self.running if not r.done)
        for req in finished:
            for c in req.cache:
                c.release()

    def run(self):
        while self.waiting or self.running:
            self.step()
