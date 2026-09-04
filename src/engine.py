from collections import deque
from src.model_runner import EOS_TOKEN, prefill, decode
from src.cache.paged_cache import make_paged_cache

class Engine:
    def __init__(self, pools, max_batch=16):
        self.waiting = deque()
        self.running = deque()
        self.pools = pools
        self.max_batch = max_batch

    def add_request(self, req):
        self.waiting.append(req)

    def step(self):
        while self.waiting and len(self.running) < self.max_batch and all(pool.has_free_blocks() for pool in self.pools):
            req = self.waiting.popleft()
            req.cache = make_paged_cache(self.pools)
            self.running.append(req)
        
        for req in self.running:
            token = prefill(req) if not req.prefilled else decode(req)
            if token == EOS_TOKEN:
                req.mark_done()
            else:
                req.yield_token(token)


        finished = [req for req in self.running if req.done]
        self.running = deque(req for req in self.running if not req.done)
        for req in finished:
            for c in req.cache:
                c.release()

    def run(self):
        while self.waiting or self.running:
            self.step()
