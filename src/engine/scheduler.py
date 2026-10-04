import time


class FifoScheduler:
    def order(self, reqs, engine):
        return list(reqs)

    def observe(self, kind, ms, units):
        pass


class PriorityScheduler(FifoScheduler):
    def __init__(self, age_weight=1.0):
        self.age_weight = age_weight
        self.cost_ms = {"piece": 20.0, "token": 0.1}

    def observe(self, kind, ms, units):
        if units:
            self.cost_ms[kind] += 0.1 * (ms / units - self.cost_ms[kind])

    def work_ms(self, req, engine):
        pieces = sum(0 if item.key is not None and engine.encode_cache.peek(item.key) is not None
                     else item.pieces - item.encoded for item in req.media)
        tokens = len(req.prompt_tokens) + len(req.output_tokens) - req.prefill_pos
        return pieces * self.cost_ms["piece"] + tokens * self.cost_ms["token"]

    def order(self, reqs, engine):
        now = time.time()
        return sorted(reqs, key=lambda r: self.work_ms(r, engine) - self.age_weight * 1000 * (now - r.arrival_time))
