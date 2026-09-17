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
