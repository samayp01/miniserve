import asyncio

import pytest

from src.model_runner import model, tokenizer
from src.runtime import GenerationError, MiniserveRuntime, StreamDetokenizer

PARTY = [9468, 236, 231]
ROCKET_WAVE = [9468, 248, 222, 9468, 234, 232]


def test_detokenizer_holds_back_partial_multibyte():
    detok = StreamDetokenizer(tokenizer)
    assert [detok.add(t) for t in PARTY] == ["", "", "\U0001F389"]


def test_detokenizer_emits_each_character_as_it_completes():
    detok = StreamDetokenizer(tokenizer)
    assert [detok.add(t) for t in ROCKET_WAVE] == ["", "", "\U0001F680", "", "", "\U0001F30A"]


def test_detokenizer_flush_returns_trailing_partial():
    detok = StreamDetokenizer(tokenizer)
    for t in PARTY[:2]:
        detok.add(t)
    assert detok.flush() == "�"


def test_detokenizer_concatenation_matches_full_decode():
    detok = StreamDetokenizer(tokenizer)
    ids = tokenizer.encode("The capital of France is Paris \U0001F680", add_special_tokens=False)
    streamed = "".join(detok.add(t) for t in ids) + detok.flush()
    assert streamed == tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def _runtime(**kwargs):
    return MiniserveRuntime(model, tokenizer, num_blocks=64, **kwargs)


def test_submit_rejects_oversized_request():
    runtime = MiniserveRuntime(model, tokenizer, num_blocks=4, block_size=16)
    with pytest.raises(ValueError):
        runtime.submit("What is the capital of France?", max_tokens=128)
    assert runtime.streams == []


def test_streams_text_then_stats():
    async def go():
        runtime = _runtime()
        async with runtime.running():
            stream = runtime.submit("What is the capital of France?", max_tokens=16)
            text = "".join([delta async for delta in stream])
        return runtime, text, stream.stats

    runtime, text, stats = asyncio.run(go())
    assert text.strip()
    assert 0 < stats["tokens"] <= 16
    assert stats["ttft_ms"] is not None
    assert stats["latency_ms"] >= stats["ttft_ms"]
    assert runtime.streams == []


def test_concurrent_streams_stay_separate():
    async def go():
        runtime = _runtime()
        async with runtime.running():
            streams = [runtime.submit(p, max_tokens=12) for p in
                       ("What is 2 + 2?", "Who wrote Hamlet?", "Name a primary color.")]
            return await asyncio.gather(*(_collect(s) for s in streams))

    async def _collect(stream):
        return "".join([delta async for delta in stream])

    texts = asyncio.run(go())
    assert len(texts) == 3
    assert all(t.strip() for t in texts)


ESSAY = "Write a detailed multi-paragraph essay about the Roman empire."


def _bounded(coro, timeout=60):
    return asyncio.run(asyncio.wait_for(coro, timeout))


async def _consume(stream):
    deltas = []
    try:
        async for delta in stream:
            deltas.append(delta)
    except GenerationError as error:
        return deltas, error
    return deltas, None


def _all_blocks_free(runtime):
    return all(sorted(pool.allocator.free) == list(range(pool.num_blocks))
               for pool in runtime.engine.pools)


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_submit_rejects_non_positive_max_tokens(max_tokens):
    runtime = _runtime()
    with pytest.raises(ValueError):
        runtime.submit("Hi", max_tokens=max_tokens)
    assert runtime.streams == []


def test_cancel_mid_stream_frees_the_request():
    async def go():
        runtime = _runtime()
        async with runtime.running():
            stream = runtime.submit(ESSAY, max_tokens=200)
            await stream.__anext__()
            stream.cancel()
            await asyncio.sleep(0.05)
        return runtime, stream.request

    runtime, req = _bounded(go())
    assert not req.done
    assert len(req.output_tokens) < 200
    assert runtime.streams == []
    assert not runtime.engine.running and not runtime.engine.waiting
    assert _all_blocks_free(runtime)


def test_step_failure_fails_running_stream_spares_waiting_and_loop_survives(fail_step):
    async def go():
        runtime = _runtime(max_batch=1)
        fail_step(runtime, on_call=3)
        async with runtime.running():
            running = runtime.submit(ESSAY, max_tokens=64)
            queued = runtime.submit("What is the capital of France?", max_tokens=16)
            results = await asyncio.gather(_consume(running), _consume(queued))
            later = await _consume(runtime.submit("Name a primary color.", max_tokens=8))
        return runtime, results, later, running, queued

    runtime, results, later, running, queued = _bounded(go())
    (running_deltas, running_error), (queued_deltas, queued_error) = results

    assert isinstance(running_error, GenerationError)
    assert "injected step failure" in str(running_error)
    assert running_deltas
    assert not running.request.done
    assert len(running.request.output_tokens) < 64
    assert queued_error is None and "".join(queued_deltas).strip()
    assert queued.stats is not None
    assert later[1] is None and "".join(later[0]).strip()
    assert runtime.streams == []
    assert _all_blocks_free(runtime)


def test_streaming_failure_is_isolated_to_its_stream():
    async def go():
        runtime = _runtime()
        async with runtime.running():
            poisoned = runtime.submit(ESSAY, max_tokens=32)
            healthy = runtime.submit("What is the capital of France?", max_tokens=16)

            def explode(token):
                raise ValueError("injected decode failure")

            poisoned._detok.add = explode
            results = await asyncio.gather(_consume(poisoned), _consume(healthy))
        return runtime, results, poisoned

    runtime, results, poisoned = _bounded(go())
    (_, poisoned_error), (healthy_deltas, healthy_error) = results

    assert isinstance(poisoned_error, GenerationError)
    assert "injected decode failure" in str(poisoned_error)
    assert not poisoned.request.done
    assert healthy_error is None and "".join(healthy_deltas).strip()
    assert runtime.streams == []
    assert _all_blocks_free(runtime)
