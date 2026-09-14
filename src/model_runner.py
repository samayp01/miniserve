import mlx.core as mx
from mlx_lm import load
from src.cache.paged_cache import BatchedPagedCache
from src.request import Request

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(MODEL_NAME)
WEIGHTS_BYTES = mx.get_active_memory()
EOS_TOKEN = tokenizer.eos_token_id

def prefill_chunk(req: Request, chunk_size: int) -> tuple[int | None, int]:
    context = req.prompt_tokens + req.output_tokens
    chunk = context[req.prefill_pos:req.prefill_pos + chunk_size]
    logits = model(mx.array(chunk)[None], cache=req.cache)
    mx.eval(logits)
    req.prefill_pos += len(chunk)
    if req.prefill_pos >= len(context):
        req.prefilled = True
        return int(mx.argmax(logits[:, -1, :], axis=-1).item()), len(chunk)
    return None, len(chunk)

def batched_decode(requests) -> list[int]:
    inputs = mx.array([[r.output_tokens[-1]] for r in requests])
    num_layers = len(requests[0].cache)
    cache = [BatchedPagedCache([r.cache[L] for r in requests]) for L in range(num_layers)]
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    return mx.argmax(logits[:, -1, :], axis=-1).tolist()
