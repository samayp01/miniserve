from abc import ABC, abstractmethod

import mlx.core as mx

class ModelAdapter(ABC):
    eos_token: int

    @abstractmethod
    def embed_tokens(self, token_ids: list[int]) -> mx.array:
        pass

    @abstractmethod
    def prefill(self, vectors: mx.array, cache) -> mx.array:
        pass

    @abstractmethod
    def decode(self, token_ids: list[list[int]], cache) -> mx.array:
        pass