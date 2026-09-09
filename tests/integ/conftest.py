import socket
import subprocess
import sys
import time

import pytest


def _wait_for_server(host, port, timeout=30):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)

    raise RuntimeError("Server did not start in time")


@pytest.fixture(scope="session")
def server():
    host = "127.0.0.1"
    port = 8000

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.server:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
    )

    try:
        _wait_for_server(host, port)
        yield f"http://{host}:{port}"
    finally:
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()