from abc import ABC, abstractmethod

import mlx.core as mx

from src.engine.request import Media

class ModelAdapter(ABC):
    eos_token: int

    @abstractmethod
    def prepare(self, prompt: str, media: list[dict] = ()) -> tuple[list[int], list[Media]]:
        pass

    @abstractmethod
    def embed_tokens(self, token_ids: list[int]) -> mx.array:
        pass

    @abstractmethod
    def prefill(self, vectors: mx.array, cache) -> mx.array:
        pass

    @abstractmethod
    def decode(self, token_ids: list[list[int]], cache) -> mx.array:
        pass

    def encode(self, media: Media, start: int, end: int) -> mx.array:
        raise NotImplementedError(f"{type(self).__name__} can't take media")
