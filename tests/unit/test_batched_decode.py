import mlx.core as mx
import pytest
from mlx_lm.models.cache import make_prompt_cache as builtin_cache

from src.model_runner import model, tokenizer, prefill_chunk, batched_decode
from src.request import Request
from src.cache.paged_cache import make_block_pools, make_paged_cache

PROMPTS = [
    "The capital of France is",
    "Water boils at one hundred degrees Celsius because the vapor pressure of the liquid",
    "Hi",
]
STEPS = 16
BLOCK_SIZE = 4


def _reference(prompt, steps):
    cache = builtin_cache(model)
    logits = model(mx.array(tokenizer.encode(prompt))[None], cache=cache)
    out = []
    for _ in range(steps):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        out.append(token.item())
        logits = model(token[None], cache=cache)
    return out


def _batched(prompts, steps):
    pools = make_block_pools(model, num_blocks=1024, block_size=BLOCK_SIZE)
    reqs = []
    for prompt in prompts:
        req = Request(tokenizer.encode(prompt), max_output_tokens=steps)
        req.cache = make_paged_cache(pools)
        token, _ = prefill_chunk(req, 1 << 20)
        req.yield_token(token)
        reqs.append(req)
    for _ in range(steps - 1):
        for req, token in zip(reqs, batched_decode(reqs)):
            req.yield_token(token)
    return reqs


@pytest.fixture(scope="module")
def batch():
    return _batched(PROMPTS, STEPS)


def test_batch_spans_ragged_block_tables(batch):
    sizes = {len(req.cache[0].block_table) for req in batch}
    assert len(sizes) > 1, f"prompts no longer produce ragged block counts: {sizes}"


def test_batched_decode_matches_single_sequence(batch):
    for prompt, req in zip(PROMPTS, batch):
        assert req.output_tokens == _reference(prompt, STEPS), prompt


def test_batch_of_one_matches_single_sequence():
    [req] = _batched(PROMPTS[:1], STEPS)
    assert req.output_tokens == _reference(PROMPTS[0], STEPS)
