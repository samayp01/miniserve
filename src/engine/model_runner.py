import mlx.core as mx
from src.cache.paged_cache import BatchedPagedCache
from src.engine.request import Request
from src.models.llama import LlamaAdapter

llama = LlamaAdapter()
model, tokenizer = llama.model, llama.tokenizer
WEIGHTS_BYTES = mx.get_active_memory()

def embed(req: Request, start: int, end: int, adapter) -> mx.array:
    context = req.prompt_tokens + req.output_tokens
    return adapter.embed_tokens(context[start:end])

def prefill_chunk(req: Request, chunk_size: int, adapter=llama) -> tuple[int | None, int]:
    context = req.prompt_tokens + req.output_tokens
    chunk = context[req.prefill_pos:req.prefill_pos + chunk_size]
    vectors = embed(req, req.prefill_pos, req.prefill_pos + len(chunk), adapter)
    logits = adapter.prefill(vectors, req.cache)
    mx.eval(logits)
    req.prefill_pos += len(chunk)
    if req.prefill_pos >= len(context):
        req.prefilled = True
        return int(mx.argmax(logits[:, -1, :], axis=-1).item()), len(chunk)
    return None, len(chunk)

def batched_decode(requests, adapter=llama) -> list[int]:
    inputs = [[r.output_tokens[-1]] for r in requests]
    num_layers = len(requests[0].cache)
    cache = [BatchedPagedCache([r.cache[L] for r in requests]) for L in range(num_layers)]
    logits = adapter.decode(inputs, cache)
    mx.eval(logits)
    return mx.argmax(logits[:, -1, :], axis=-1).tolist()
