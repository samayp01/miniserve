import signal
from contextlib import contextmanager

import mlx.core as mx
import pytest

from src.engine.model_runner import tokenizer
from src.engine.engine import Engine
from src.engine.request import Media, Request
from src.cache.paged_cache import make_block_pools
from tests.unit.fakes import FakeAdapter

BLOCKS, BLOCK_SIZE = 4, 16
FAKE = FakeAdapter()


def _engine(num_blocks=BLOCKS, block_size=BLOCK_SIZE, **kwargs):
    return Engine(make_block_pools(FAKE, num_blocks, block_size), adapter=FAKE, **kwargs)


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
    pools = make_block_pools(FAKE, num_blocks, BLOCK_SIZE)
    engine = Engine(pools, chunk_size=chunk_size, adapter=FAKE)
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


@pytest.mark.parametrize("chunk_size", [8, 64])
def test_chunked_prefill_under_pressure_completes_and_returns_every_block(chunk_size):
    num_blocks = 8
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
    pools = make_block_pools(FAKE, 32, BLOCK_SIZE)
    engine = Engine(pools, adapter=FAKE)
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
    pools = make_block_pools(FAKE, 16, BLOCK_SIZE)
    engine = Engine(pools, adapter=FAKE)
    req = Request(_chat("Say hi."), max_output_tokens=8)
    engine.add_request(req)
    with time_limit(60):
        engine.run()

    engine.abort(req)
    engine.abort(req)

    for pool in pools:
        assert sorted(pool.allocator.free) == list(range(16))


def test_timestamps_follow_the_stages_in_order():
    engine = _engine(num_blocks=64)
    req = Request(list(range(20)), max_output_tokens=4)
    engine.add_request(req)
    engine.run()
    assert req.arrival_time <= req.admitted_time <= req.encoded_time <= req.first_token_time <= req.finish_time


def _media_request(pieces=17, per_piece=2):
    positions = list(range(5, 5 + pieces * per_piece))
    media = Media(positions=positions, data=mx.zeros((pieces, 1)))
    return Request(list(range(5 + pieces * per_piece + 5)), max_output_tokens=4, media=[media])


def test_encode_budget_spreads_pieces_across_steps():
    fake = FakeAdapter()
    engine = Engine(make_block_pools(fake, 64, BLOCK_SIZE), adapter=fake, encode_budget=4)
    req = _media_request()
    engine.add_request(req)
    for _ in range(4):
        engine.step()
        assert not req.prefilled
    engine.step()
    assert fake.encode_calls == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 17)]
    assert req.prefilled
    assert len(req.media[0].embeds) == len(req.media[0].positions)


def test_encode_budget_is_shared_across_requests():
    fake = FakeAdapter()
    engine = Engine(make_block_pools(fake, 64, BLOCK_SIZE), adapter=fake, encode_budget=4)
    first, second = _media_request(pieces=3), _media_request(pieces=3)
    engine.add_request(first)
    engine.add_request(second)
    engine.step()
    assert fake.encode_calls == [(0, 3), (0, 1)]
    assert first.media[0].ready and second.media[0].encoded == 1


def test_preemption_does_not_re_encode():
    fake = FakeAdapter()
    engine = Engine(make_block_pools(fake, 64, BLOCK_SIZE), adapter=fake, encode_budget=4)
    req = _media_request()
    engine.add_request(req)
    _step_until(engine, lambda: req.prefilled)
    calls = list(fake.encode_calls)
    engine._preempt(req)
    with time_limit(60):
        engine.run()
    assert req.done
    assert fake.encode_calls == calls


def test_media_is_marked_encoded_only_after_its_last_piece():
    fake = FakeAdapter()
    engine = Engine(make_block_pools(fake, 64, BLOCK_SIZE), adapter=fake, encode_budget=4)
    req = _media_request()
    engine.add_request(req)
    for _ in range(4):
        engine.step()
        assert req.encoded_time is None
    engine.step()
    assert req.admitted_time < req.encoded_time
