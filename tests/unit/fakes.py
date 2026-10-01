import mlx.core as mx

from src.models.base import ModelAdapter


class FakeAdapter(ModelAdapter):
    eos_token = 0

    def __init__(self, num_layers=2):
        self.layers = [None] * num_layers
        self.encode_calls = []

    def prepare(self, prompt, media=()):
        return list(range(1, len(prompt) + 1)), []

    def encode(self, media, start, end):
        self.encode_calls.append((start, end))
        per_piece = len(media.positions) // media.pieces
        return mx.full(((end - start) * per_piece, 1), 7.0)

    def embed_tokens(self, token_ids):
        return mx.zeros((1, len(token_ids), 1))

    def prefill(self, vectors, cache):
        self._store(cache, vectors.shape[1])
        return self._logits(1, vectors.shape[1])

    def decode(self, token_ids, cache):
        self._store([c for layer in cache for c in layer.caches], 1)
        return self._logits(len(token_ids), 1)

    def _store(self, caches, n):
        kv = mx.zeros((1, 1, n, 1))
        for c in caches:
            c.store(kv, kv)
        mx.eval([c.pool.k_pool for c in caches] + [c.pool.v_pool for c in caches])

    def _logits(self, batch, length):
        return mx.broadcast_to(mx.array([0.0, 1.0]), (batch, length, 2))
