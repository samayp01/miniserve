import signal

import pytest

from tests.llama import llama, tokenizer
from src.engine.engine import Engine
from src.engine.request import Request
from src.cache.paged_cache import make_block_pools
from src.cache.swap import NotEnoughBlocks
from tests.unit.fakes import FakeAdapter

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
FAKE = FakeAdapter()


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


def _build(num_blocks, adapter, **engine_kwargs):
    pools = make_block_pools(adapter, num_blocks, BLOCK_SIZE)
    engine = Engine(pools, max_batch=8, adapter=adapter, **engine_kwargs)
    reqs = []
    for prompt in PROMPTS:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        req = CountingRequest(ids, max_output_tokens=MAX_OUT)
        engine.add_request(req)
        reqs.append(req)
    return engine, pools, reqs


def _drive(engine):
    previous = signal.signal(signal.SIGALRM, _expired)
    signal.alarm(RUN_TIMEOUT)
    try:
        engine.run()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _run(num_blocks, adapter, **engine_kwargs):
    engine, pools, reqs = _build(num_blocks, adapter, **engine_kwargs)
    _drive(engine)
    return engine, pools, reqs


@pytest.fixture(scope="module")
def roomy():
    return _run(ROOMY_BLOCKS, FAKE)


@pytest.fixture(scope="module")
def tight():
    return _run(TIGHT_BLOCKS, FAKE)


@pytest.fixture(scope="module")
def roomy_real():
    return _run(ROOMY_BLOCKS, llama)


@pytest.fixture(scope="module")
def tight_real():
    return _run(TIGHT_BLOCKS, llama)


def test_roomy_pool_never_preempts(roomy):
    engine, _, _ = roomy
    assert engine.preemptions == 0


def test_tight_pool_preempts(tight):
    engine, _, _ = tight
    assert engine.preemptions > 0


def test_preempted_output_matches_unpreempted(tight_real, roomy_real):
    _, _, tight_reqs = tight_real
    _, _, roomy_reqs = roomy_real
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


@pytest.fixture(scope="module")
def tight_swap_real():
    return _run(TIGHT_BLOCKS, llama, swap_min_tokens=0)


def test_swapped_output_matches_unpreempted(tight_swap_real, roomy_real):
    engine, _, swapped = tight_swap_real
    _, _, clean = roomy_real
    assert engine.swaps > 0
    for prompt, a, b in zip(PROMPTS, swapped, clean):
        assert a.output_tokens == b.output_tokens, prompt


def _prompt_tokens(reqs):
    return sum(len(r.prompt_tokens) for r in reqs)


def _all_free(pools):
    return all(sorted(p.allocator.free) == list(range(TIGHT_BLOCKS)) for p in pools)


def _no_swap_files(engine):
    return engine.swap.bytes == 0 and engine.swap.meta == {} and not any(engine.swap.directory.iterdir())


def test_swap_never_prefills_a_context_twice():
    fake = FakeAdapter()
    engine, pools, reqs = _run(TIGHT_BLOCKS, fake, swap_min_tokens=0)
    assert engine.preemptions > 0 and engine.swaps == engine.preemptions
    assert fake.prefilled_tokens == _prompt_tokens(reqs)
    assert all(r.done for r in reqs) and _all_free(pools) and _no_swap_files(engine)


def test_recompute_prefills_preempted_contexts_again():
    fake = FakeAdapter()
    engine, _, reqs = _run(TIGHT_BLOCKS, fake)
    assert engine.preemptions > 0
    assert fake.prefilled_tokens > _prompt_tokens(reqs)


def test_auto_threshold_above_every_context_never_swaps():
    engine, pools, reqs = _run(TIGHT_BLOCKS, FakeAdapter(), swap_min_tokens=10**6)
    assert engine.preemptions > 0 and engine.swaps == 0
    assert all(r.done for r in reqs) and _all_free(pools)


def test_failed_load_keeps_the_request_waiting_and_the_engine_running():
    engine, pools, reqs = _build(TIGHT_BLOCKS, FakeAdapter(), swap_min_tokens=0)
    real_load, failures = engine.swap.load, []

    def flaky(key, caches):
        if not failures:
            failures.append(key)
            raise NotEnoughBlocks("injected")
        return real_load(key, caches)

    engine.swap.load = flaky
    _drive(engine)
    assert failures
    assert all(r.done for r in reqs) and _all_free(pools) and _no_swap_files(engine)


def test_aborting_a_swapped_request_deletes_its_file():
    engine, pools, reqs = _build(TIGHT_BLOCKS, FakeAdapter(), swap_min_tokens=0)
    swapped = []
    for _ in range(500):
        engine.step()
        swapped = [r for r in reqs if r.swapped]
        if swapped:
            break
    victim = swapped[0]
    engine.abort(victim)
    assert f"req{victim.id}" not in engine.swap.meta
    assert not (engine.swap.directory / f"req{victim.id}.npy").exists()
    _drive(engine)
    assert all(r.done for r in reqs if r is not victim)
    assert _all_free(pools) and _no_swap_files(engine)
