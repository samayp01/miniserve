import pytest


@pytest.fixture
def fail_step():
    def install(runtime, on_call):
        real_step = runtime.engine.step
        calls = 0

        def step():
            nonlocal calls
            calls += 1
            if calls == on_call:
                raise RuntimeError("injected step failure")
            return real_step()

        runtime.engine.step = step

    return install
