import signal

import pytest

from src.model_runner import model, tokenizer
from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools

PROMPTS = [
    "Write a detailed multi-paragraph essay about the Roman empire.",
    "Write a long detailed explanation of how photosynthesis works.",
    "Write several paragraphs describing the history of the ocean.",
]
MAX_OUT = 64
BLOCK_SIZE = 16
TIGHT_BLOCKS = 14
ROOMY_BLOCKS = 64
RUN_TIMEOUT = 60


class CountingRequest(Request):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yields = 0

    def yield_token(self, token):
        if token is not None:
            self.yields += 1
        super().yield_token(token)


def _expired(signum, frame):
    raise TimeoutError(f"engine.run() did not return within {RUN_TIMEOUT}s")


def _run(num_blocks):
    pools = make_block_pools(model, num_blocks, BLOCK_SIZE)
    engine = Engine(pools, max_batch=8)
    reqs = []
    for prompt in PROMPTS:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        req = CountingRequest(ids, max_output_tokens=MAX_OUT)
        engine.add_request(req)
        reqs.append(req)

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.alarm(RUN_TIMEOUT)
    try:
        engine.run()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return engine, pools, reqs


@pytest.fixture(scope="module")
def roomy():
    return _run(ROOMY_BLOCKS)


@pytest.fixture(scope="module")
def tight():
    return _run(TIGHT_BLOCKS)


def test_roomy_pool_never_preempts(roomy):
    engine, _, _ = roomy
    assert engine.preemptions == 0


def test_tight_pool_preempts(tight):
    engine, _, _ = tight
    assert engine.preemptions > 0


def test_preempted_output_matches_unpreempted(tight, roomy):
    _, _, tight_reqs = tight
    _, _, roomy_reqs = roomy
    for prompt, preempted, clean in zip(PROMPTS, tight_reqs, roomy_reqs):
        assert preempted.output_tokens == clean.output_tokens, prompt


def test_preemption_recomputes_kv_without_regenerating_tokens(tight):
    _, _, reqs = tight
    assert all(req.done for req in reqs)
    assert [req.yields for req in reqs] == [MAX_OUT] * len(PROMPTS)


def test_preemption_returns_every_block_exactly_once(tight):
    _, pools, _ = tight
    for pool in pools:
        assert sorted(pool.allocator.free) == list(range(TIGHT_BLOCKS))
