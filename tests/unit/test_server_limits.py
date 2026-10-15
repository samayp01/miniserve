import pytest
from fastapi.testclient import TestClient

import src.api.server as server
from src.api.runtime import MiniserveRuntime
from tests.llama import llama

GARBAGE = {"content": b"not json", "headers": {"content-type": "application/json"}}


@pytest.fixture
def served(monkeypatch):
    runtime = MiniserveRuntime(llama, num_blocks=64)
    monkeypatch.setattr(server, "runtime", runtime)
    with TestClient(server.app) as client:
        yield client, runtime


def test_generate_is_rejected_before_its_body_is_read(served):
    client, runtime = served
    runtime.max_waiting = 0
    response = client.post("/generate", **GARBAGE)
    assert response.status_code == 503 and response.headers["retry-after"] == "1"
    assert runtime.rejected == 1


def test_frame_is_rejected_before_its_body_is_read(served):
    client, runtime = served
    runtime.max_preparing = 0
    stream_id = client.post("/streams", json={"fps": 1}).json()["stream_id"]
    response = client.post(f"/streams/{stream_id}/frames", **GARBAGE)
    assert response.status_code == 503
    assert runtime.media_streams[stream_id].dropped == 1


def test_invalid_generate_body_is_still_a_validation_error(served):
    client, _ = served
    assert client.post("/generate", json={"prompt": "Hi", "max_tokens": 0}).status_code == 422
