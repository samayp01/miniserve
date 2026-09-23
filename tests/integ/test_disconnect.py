import asyncio
import json

import httpx
import uvicorn

import src.server as server
from src.model_runner import model, tokenizer
from src.runtime import MiniserveRuntime

ESSAY = {"prompt": "Write a long essay about the Roman empire.", "max_tokens": 400}
BLOCKS = 256


async def _serve():
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.01)
    port = srv.servers[0].sockets[0].getsockname()[1]
    return srv, task, f"http://127.0.0.1:{port}/generate"


async def _until(condition, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _run(scenario, runtime, monkeypatch, timeout=60):
    monkeypatch.setattr(server, "runtime", runtime)

    async def go():
        srv, task, url = await _serve()
        try:
            return await scenario(url)
        finally:
            srv.should_exit = True
            await task

    return asyncio.run(asyncio.wait_for(go(), timeout))


def _all_blocks_free(runtime):
    return all(sorted(pool.allocator.free) == list(range(BLOCKS)) for pool in runtime.engine.pools)


def test_disconnect_mid_stream_aborts_the_request(monkeypatch):
    runtime = MiniserveRuntime(model, tokenizer, num_blocks=BLOCKS)

    async def scenario(url):
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream("POST", url, json=ESSAY) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        req = runtime.streams[0].request
                        break
        drained = await _until(lambda: not runtime.streams and not runtime.engine.running)
        return drained, req

    drained, req = _run(scenario, runtime, monkeypatch)

    assert drained
    assert not req.done
    assert len(req.output_tokens) < ESSAY["max_tokens"]
    assert _all_blocks_free(runtime)


def test_disconnect_while_queued_removes_the_request_without_running_it(monkeypatch):
    runtime = MiniserveRuntime(model, tokenizer, num_blocks=BLOCKS, max_batch=1)

    async def scenario(url):
        async with httpx.AsyncClient(timeout=30) as keeper, httpx.AsyncClient(timeout=30) as leaver:
            async def keep():
                frames = []
                async with keeper.stream("POST", url, json={**ESSAY, "max_tokens": 120}) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            frames.append(json.loads(line.removeprefix("data:")))
                return frames

            kept = asyncio.create_task(keep())
            assert await _until(lambda: len(runtime.engine.running) == 1)
            async with leaver.stream("POST", url, json=ESSAY):
                assert await _until(lambda: len(runtime.engine.waiting) == 1)
                queued = runtime.engine.waiting[0]
            left = await _until(lambda: not runtime.engine.waiting)
            return left, queued, await kept

    left, queued, kept_frames = _run(scenario, runtime, monkeypatch)

    assert left
    assert queued.output_tokens == [] and queued.cache is None and not queued.done
    assert kept_frames[-1].get("done") is True
    assert _all_blocks_free(runtime)
