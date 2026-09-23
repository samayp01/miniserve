import asyncio
import json

from src.model_runner import model, tokenizer
from src.runtime import MiniserveRuntime
from src.server import events

ESSAY = "Write a detailed multi-paragraph essay about the Roman empire."


def _runtime():
    return MiniserveRuntime(model, tokenizer, num_blocks=64)


async def _connected():
    return False


async def _collect(generator):
    return [chunk async for chunk in generator]


def _frames(chunks):
    assert all(chunk.startswith("data: ") and chunk.endswith("\n\n") for chunk in chunks)
    return [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]


def _bounded(coro, timeout=60):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_streams_delta_frames_then_a_done_frame():
    async def go():
        runtime = _runtime()
        async with runtime.running():
            stream = runtime.submit("What is the capital of France?", max_tokens=16)
            chunks = await _collect(events(stream, _connected))
        return runtime, chunks

    runtime, chunks = _bounded(go())
    frames = _frames(chunks)

    assert len(frames) > 1
    assert all(set(frame) == {"delta"} for frame in frames[:-1])
    assert frames[-1]["done"] is True and frames[-1]["tokens"] > 0
    assert runtime.streams == []


def test_generation_failure_ends_with_an_error_frame(fail_step):
    async def go():
        runtime = _runtime()
        fail_step(runtime, on_call=2)
        async with runtime.running():
            stream = runtime.submit(ESSAY, max_tokens=64)
            chunks = await _collect(events(stream, _connected))
        return runtime, chunks, stream.request

    runtime, chunks, req = _bounded(go())
    frames = _frames(chunks)

    assert frames[-1] == {"error": "injected step failure"}
    assert not req.done
    assert not runtime.engine.running and not runtime.engine.waiting
    assert not any(frame.get("done") for frame in frames)
    assert runtime.streams == []


def test_stops_and_cancels_the_request_once_the_client_disconnects():
    async def go():
        runtime = _runtime()
        checks = 0

        async def gone_after_two_deltas():
            nonlocal checks
            checks += 1
            return checks > 2

        async with runtime.running():
            stream = runtime.submit(ESSAY, max_tokens=200)
            chunks = await _collect(events(stream, gone_after_two_deltas))
        return runtime, chunks, stream.request

    runtime, chunks, req = _bounded(go())
    frames = _frames(chunks)

    assert len(frames) == 2 and all("delta" in frame for frame in frames)
    assert not req.done
    assert runtime.streams == []
    assert not runtime.engine.running and not runtime.engine.waiting
    assert all(sorted(pool.allocator.free) == list(range(64)) for pool in runtime.engine.pools)
