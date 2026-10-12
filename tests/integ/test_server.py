import base64
import io
import json

import httpx
import pytest
from PIL import Image


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


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_generate_rejects_non_positive_max_tokens(server, max_tokens):
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{server}/generate",
            json={"prompt": "Hello", "max_tokens": max_tokens},
        )

    assert response.status_code == 422


def test_generate_rejects_media_the_model_cannot_take(server):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    image = base64.b64encode(buffer.getvalue()).decode()
    with httpx.Client(timeout=60) as client:
        response = client.post(f"{server}/generate", json={"prompt": "Hi", "media": [{"type": "image", "data": image}]})
    assert response.status_code == 400
    assert "can't take media" in response.json()["detail"]


def _png():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_stream_endpoints_on_a_model_without_media(server):
    with httpx.Client(timeout=60) as client:
        stream_id = client.post(f"{server}/streams", json={"fps": 1, "window": 4}).json()["stream_id"]
        frame = client.post(f"{server}/streams/{stream_id}/frames", json={"data": _png()})
        assert frame.status_code == 400 and "can't take media" in frame.json()["detail"]
        assert client.post(f"{server}/streams/{stream_id}/ask", json={"prompt": "What now?"}).status_code == 400
        assert client.post(f"{server}/streams/nope/frames", json={"data": _png()}).status_code == 404
        assert client.delete(f"{server}/streams/{stream_id}").json() == {"kept": 0, "skipped": 0, "dropped": 0}
        assert client.delete(f"{server}/streams/{stream_id}").status_code == 404
