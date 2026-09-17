import json

import httpx


def test_generate(server):
    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST",
            f"{server}/generate",
            json={
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
            },
        ) as response:
            assert response.status_code == 200

            text = ""
            done = None

            for line in response.iter_lines():
                if not line:
                    continue

                assert line.startswith("data: ")

                event = json.loads(line[6:])

                if event.get("done"):
                    done = event
                else:
                    text += event["delta"]

    assert text
    assert done is not None
    assert done["tokens"] > 0
    assert done["ttft_ms"] is not None
    assert done["latency_ms"] is not None


def test_generate_rejects_request_larger_than_cache(server):
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{server}/generate",
            json={"prompt": "Hello", "max_tokens": 100_000},
        )

    assert response.status_code == 400
    assert "blocks" in response.json()["detail"]
