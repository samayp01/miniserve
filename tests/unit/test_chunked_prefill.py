import mlx.core as mx
from src.model_runner import model, tokenizer
from src.cache.paged_cache import make_block_pools, make_paged_cache

PROMPT = (
    "The capital of France is Paris. The Eiffel Tower was completed in 1889 "
    "for the World's Fair and stands about 330 meters tall. Tell me why it "
    "remains one of the most visited monuments in the world."
)

def _run(ids, n=8, chunk=None):
    pools = make_block_pools(model, num_blocks=128, block_size=16)
    cache = make_paged_cache(pools)
    if chunk is None:
        logits = model(mx.array(ids)[None], cache=cache)
    else:
        for i in range(0, len(ids), chunk):
            logits = model(mx.array(ids[i:i + chunk])[None], cache=cache)
    out = []
    for _ in range(n):
        t = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(t.item())
        logits = model(t[None], cache=cache)
    return out

def test_chunked_prefill_matches_full():
    ids = tokenizer.encode(PROMPT)
    full = _run(ids)
    print(f"\nprompt tokens={len(ids)}  full-prefill greedy={full}")
    for chunk in (16, 24, 8):
        chunked = _run(ids, chunk=chunk)
        print(f"chunk={chunk:>2}  greedy={chunked}  match={chunked == full}")
        assert chunked == full, f"chunk={chunk} diverged from full prefill"
