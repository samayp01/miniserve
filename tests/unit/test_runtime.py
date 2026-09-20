import asyncio

import pytest

from src.model_runner import model, tokenizer
from src.runtime import MiniserveRuntime, StreamDetokenizer

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
