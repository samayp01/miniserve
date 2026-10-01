import mlx.core as mx
from mlx_lm import load

from src.models.base import ModelAdapter

LLAMA_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"


class LlamaAdapter(ModelAdapter):
    def __init__(self, name=LLAMA_NAME):
        self.model, self.tokenizer = load(name)
        self.eos_token = self.tokenizer.eos_token_id

    @property
    def layers(self):
        return self.model.layers

    def embed_tokens(self, token_ids):
        return self.model.model.embed_tokens(mx.array(token_ids))[None]

    def prefill(self, vectors, cache):
        return self.model(None, cache=cache, input_embeddings=vectors)

    def decode(self, token_ids, cache):
        return self.model(mx.array(token_ids), cache=cache)
