import asyncio
import json

import httpx


async def _generate(client, server, prompt):
    text = ""
    done = None

    async with client.stream(
        "POST",
        f"{server}/generate",
        json={
            "prompt": prompt,
            "max_tokens": 32,
        },
    ) as response:
        assert response.status_code == 200

        async for line in response.aiter_lines():
            if not line:
                continue

            assert line.startswith("data: ")

            event = json.loads(line[6:])

            if event.get("done"):
                done = event
            else:
                text += event["delta"]

    return text, done


def test_concurrent_generation(server):
    async def run():
        async with httpx.AsyncClient(timeout=60) as client:
            results = await asyncio.gather(
                _generate(client, server, "What is 2 + 2?"),
                _generate(client, server, "What is the capital of France?"),
                _generate(client, server, "Who wrote Hamlet?"),
                _generate(client, server, "Write a haiku about the space."),
            )

        return results

    results = asyncio.run(run())

    assert len(results) == 4

    for text, done in results:
        assert text
        assert done is not None
        assert done["tokens"] > 0
        assert done["ttft_ms"] is not None
        assert done["latency_ms"] is not None