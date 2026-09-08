import mlx.core as mx
from mlx_lm import load
from src.cache.paged_cache import BatchedPagedCache
from src.request import Request

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

model, tokenizer = load(MODEL_NAME)
WEIGHTS_BYTES = mx.get_active_memory()
EOS_TOKEN = tokenizer.eos_token_id

def prefill_request(req: Request) -> int:
    context = req.prompt_tokens + req.output_tokens
    logits = model(mx.array(context)[None], cache=req.cache)
    mx.eval(logits)
    req.prefilled = True
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())

def batched_decode(requests) -> list[int]:
    inputs = mx.array([[r.output_tokens[-1]] for r in requests])
    num_layers = len(requests[0].cache)
    cache = [BatchedPagedCache([r.cache[L] for r in requests]) for L in range(num_layers)]
    logits = model(inputs, cache=cache)
    mx.eval(logits)
    return mx.argmax(logits[:, -1, :], axis=-1).tolist()
