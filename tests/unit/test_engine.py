import signal
from contextlib import contextmanager

import pytest

from src.model_runner import model, tokenizer
from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools

BLOCKS, BLOCK_SIZE = 4, 16


def _engine(num_blocks=BLOCKS, block_size=BLOCK_SIZE, **kwargs):
    return Engine(make_block_pools(model, num_blocks, block_size), **kwargs)


def test_rejects_prompt_larger_than_pool():
    engine = _engine()
    with pytest.raises(ValueError, match="pool holds 4"):
        engine.add_request(Request(list(range(100)), max_output_tokens=4))
    assert not engine.waiting


def test_rejects_when_growth_exceeds_pool():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.add_request(Request(list(range(40)), max_output_tokens=40))
    assert not engine.waiting


def test_accepts_request_that_exactly_fills_pool():
    engine = _engine()
    engine.add_request(Request(list(range(60)), max_output_tokens=4))
    assert len(engine.waiting) == 1


def test_rejects_one_token_past_capacity():
    engine = _engine()
    with pytest.raises(ValueError):
        engine.add_request(Request(list(range(61)), max_output_tokens=4))


@contextmanager
def time_limit(seconds):
    """Fail instead of hanging the suite if run() regresses into a spin."""
    def _timeout(signum, frame):
        raise TimeoutError(f"engine.run() did not return within {seconds}s")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_run_terminates():
    engine = _engine(num_blocks=64)
    req = Request(tokenizer.encode("Hi"), max_output_tokens=4)
    engine.add_request(req)

    with time_limit(120):
        engine.run()

    assert req.done
    assert len(req.output_tokens) == 4


def _drain(prompt_lengths, num_blocks, chunk_size):
    pools = make_block_pools(model, num_blocks, BLOCK_SIZE)
    engine = Engine(pools, chunk_size=chunk_size)
    reqs = [Request(list(range(1, n + 1)), max_output_tokens=8) for n in prompt_lengths]
    for req in reqs:
        engine.add_request(req)
    with time_limit(120):
        engine.run()
    return pools, reqs


def test_admission_does_not_overcommit_blocks_reserved_by_chunked_prefill():
    pools, reqs = _drain((100, 40, 40), num_blocks=10, chunk_size=16)
    assert all(req.done for req in reqs)
    for pool in pools:
        assert sorted(pool.allocator.free) == list(range(10))


@pytest.mark.parametrize("num_blocks", [8, 12, 16])
@pytest.mark.parametrize("chunk_size", [8, 16, 64])
def test_chunked_prefill_under_pressure_completes_and_returns_every_block(num_blocks, chunk_size):
    pools, reqs = _drain((100, 40, 40, 70, 20), num_blocks, chunk_size)
    assert all(req.done for req in reqs)
    for pool in pools:
        assert sorted(pool.allocator.free) == list(range(num_blocks))


ESSAY = "Write a detailed multi-paragraph essay about the Roman empire."


def _chat(prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)


def _step_until(engine, condition, limit=500):
    for _ in range(limit):
        if condition():
            return
        engine.step()
    raise AssertionError("condition never reached")


@pytest.mark.parametrize("max_output_tokens", [0, -5])
def test_rejects_non_positive_max_output_tokens(max_output_tokens):
    engine = _engine()
    with pytest.raises(ValueError, match="at least 1"):
        engine.add_request(Request([1, 2, 3], max_output_tokens=max_output_tokens))
    assert not engine.waiting


def test_abort_removes_waiting_request_before_it_runs():
    engine = _engine(num_blocks=64, max_batch=1)
    first = Request(_chat("Say hi."), max_output_tokens=8)
    second = Request(_chat("Say hi."), max_output_tokens=8)
    engine.add_request(first)
    engine.add_request(second)
    engine.step()
    assert list(engine.waiting) == [second]

    engine.abort(second)
    with time_limit(60):
        engine.run()

    assert first.done
    assert not second.done
    assert second.output_tokens == [] and second.cache is None


def test_abort_running_request_returns_its_blocks_and_spares_the_rest():
    pools = make_block_pools(model, 32, BLOCK_SIZE)
    engine = Engine(pools)
    victim = Request(_chat(ESSAY), max_output_tokens=64)
    others = [Request(_chat("Name a primary color."), max_output_tokens=16) for _ in range(2)]
    for req in [victim, *others]:
        engine.add_request(req)
    _step_until(engine, lambda: len(victim.output_tokens) >= 4)

    engine.abort(victim)
    assert victim not in engine.running and victim.cache is None
    with time_limit(60):
        engine.run()

    assert not victim.done
    assert all(req.done for req in others)
    for pool in pools:
        assert sorted(pool.allocator.free) == list(range(32))


def test_abort_after_completion_does_not_double_free():
    pools = make_block_pools(model, 16, BLOCK_SIZE)
    engine = Engine(pools)
    req = Request(_chat("Say hi."), max_output_tokens=8)
    engine.add_request(req)
    with time_limit(60):
        engine.run()

    engine.abort(req)
    engine.abort(req)

    for pool in pools:
        assert sorted(pool.allocator.free) == list(range(16))
