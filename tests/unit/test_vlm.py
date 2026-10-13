import asyncio

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.api.runtime import MiniserveRuntime
from src.cache.paged_cache import make_block_pools
from src.engine.engine import Engine
from src.engine.request import Request
from src.models.smolvlm import SmolVLMAdapter

vlm = SmolVLMAdapter()


def _circle():
    image = Image.new("RGB", (512, 512), (30, 60, 160))
    ImageDraw.Draw(image).ellipse((50, 50, 250, 300), fill=(220, 30, 30))
    return image


def _image_request(question):
    ids, media = vlm.prepare(question, [{"type": "image", "data": _circle()}])
    return Request(ids, max_output_tokens=30, media=media)


def _run(*requests, max_batch=8):
    engine = Engine(make_block_pools(vlm, 512, 16), max_batch=max_batch, adapter=vlm)
    for req in requests:
        engine.add_request(req)
    engine.run()
    return [req.output_tokens for req in requests]


def test_prepare_reserves_one_position_per_image_vector():
    ids, [image] = vlm.prepare("Describe it.", [{"type": "image", "data": _circle()}])
    assert isinstance(image.data, np.ndarray)
    assert image.data.shape[0] == 17
    assert len(image.positions) == 64 * 17
    assert all(ids[p] == vlm.image_token for p in image.positions)


def test_encode_returns_one_vector_per_position():
    _, [image] = vlm.prepare("Describe it.", [{"type": "image", "data": _circle()}])
    assert vlm.encode(image, 0, image.pieces).shape == (len(image.positions), vlm.model.config.text_config.hidden_size)


def test_answers_about_the_image_through_the_engine():
    [tokens] = _run(_image_request("What shape is in this image, and what color is it?"))
    answer = vlm.processor.tokenizer.decode(tokens).lower()
    assert "red" in answer and "circle" in answer


def test_batched_decode_matches_one_at_a_time():
    prompts = ["Write one long sentence about the history of the Roman empire and its roads.", "Hi"]
    batched = _run(*[Request(vlm.prepare(p)[0], max_output_tokens=12) for p in prompts])
    alone = [_run(Request(vlm.prepare(p)[0], max_output_tokens=12), max_batch=1)[0] for p in prompts]
    assert batched == alone


def test_rejects_unsupported_media():
    with pytest.raises(NotImplementedError, match="can't take audio"):
        vlm.prepare("Hi", [{"type": "audio", "data": None}])


def test_runtime_streams_an_answer_about_an_image():
    async def go():
        runtime = MiniserveRuntime(vlm, num_blocks=256)
        async with runtime.running():
            stream = runtime.submit("What shape is in this image, and what color is it?", 30, [{"type": "image", "data": _circle()}])
            return "".join([delta async for delta in stream]).lower()

    answer = asyncio.run(go())
    assert "red" in answer and "circle" in answer


def test_same_image_twice_hits_the_encode_cache():
    engine = Engine(make_block_pools(vlm, 512, 16), vlm)
    question = "What shape is in this image, and what color is it?"
    first, second = _image_request(question), _image_request(question)
    engine.add_request(first)
    engine.run()
    engine.add_request(second)
    engine.run()
    assert not first.media[0].hit and second.media[0].hit
    for req in (first, second):
        answer = vlm.tokenizer.decode(req.output_tokens).lower()
        assert "red" in answer and "circle" in answer


def _frame(x):
    image = Image.new("RGB", (512, 512), (30, 60, 160))
    ImageDraw.Draw(image).ellipse((50 + x, 150, 250 + x, 350), fill=(220, 30, 30))
    return image


def test_prepare_splits_several_tiled_images():
    wide = Image.new("RGB", (1024, 768), (0, 120, 0))
    ids, items = vlm.prepare("Compare them.", [{"type": "image", "data": _circle()}, {"type": "image", "data": wide}])
    assert [item.pieces for item in items] == [17, 13]
    assert [len(item.positions) for item in items] == [1088, 832]
    assert all(ids[p] == vlm.image_token for item in items for p in item.positions)
    assert items[0].key != items[1].key


def test_frames_are_untiled_and_keyed_separately():
    ids, items = vlm.prepare("What changed?", [{"type": "frame", "data": _frame(x)} for x in (0, 100)])
    assert [item.pieces for item in items] == [1, 1]
    assert [len(item.positions) for item in items] == [64, 64]
    assert all(ids[p] == vlm.image_token for item in items for p in item.positions)
    assert items[0].key != items[1].key


def test_rejects_mixing_images_and_frames():
    with pytest.raises(NotImplementedError, match="can't mix"):
        vlm.prepare("Hi", [{"type": "image", "data": _circle()}, {"type": "frame", "data": _frame(0)}])


def test_answers_about_frames_through_the_engine():
    ids, media = vlm.prepare("What color is the circle?", [{"type": "frame", "data": _frame(x)} for x in (0, 100, 200)])
    [tokens] = _run(Request(ids, max_output_tokens=20, media=media))
    assert "red" in vlm.tokenizer.decode(tokens).lower()


def test_stream_window_answers_from_prefetched_frames():
    async def go():
        runtime = MiniserveRuntime(vlm, num_blocks=256)
        async with runtime.running():
            stream = runtime.open_stream(fps=100, window=2)
            for i, x in enumerate((0, 100, 200)):
                assert stream.admit(float(i)) is None
                frame = _frame(x)
                runtime.ingest(stream, frame, vlm.prepare("", [{"type": "frame", "data": frame}])[1])
                while stream.pending is not None and not stream.pending.ready:
                    await asyncio.sleep(0.01)
            response = runtime.submit("What color is the circle?", 20, stream.media())
            text = "".join([delta async for delta in response]).lower()
            return text, response.request

    text, req = asyncio.run(go())
    assert "red" in text
    assert [item.hit for item in req.media] == [True, True]


def test_layout_matches_prepare_without_any_pixels():
    frames = [{"type": "frame", "data": _frame(x)} for x in (0, 100, 200)]
    ids, items = vlm.prepare("What is moving?", frames)
    layout_ids, layout_items = vlm.layout("What is moving?", [item.key for item in items])
    assert layout_ids == ids
    assert [i.positions for i in layout_items] == [i.positions for i in items]
    assert all(i.pieces == 1 and i.data is None for i in layout_items)


def test_attach_cached_needs_every_frame_in_the_cache():
    runtime = MiniserveRuntime(vlm, num_blocks=64)
    _, items = vlm.layout("Hi", ["a", "b"])
    runtime.engine.encode_cache.put("a", mx.zeros((64, 4)))
    assert not runtime.attach_cached(items)
    assert not any(i.ready for i in items)
    runtime.engine.encode_cache.put("b", mx.zeros((64, 4)))
    assert runtime.attach_cached(items)
    assert all(i.ready and i.hit for i in items)


def test_question_over_cached_frames_holds_no_pixels():
    async def go():
        runtime = MiniserveRuntime(vlm, num_blocks=256)
        async with runtime.running():
            stream = runtime.open_stream(fps=100, window=3)
            for i, x in enumerate((0, 100, 200)):
                assert stream.admit(float(i)) is None
                frame = _frame(x)
                runtime.ingest(stream, frame, vlm.prepare("", [{"type": "frame", "data": frame}])[1])
                while stream.pending is not None and not stream.pending.ready:
                    await asyncio.sleep(0.01)
            ids, items = vlm.layout("What color is the circle?", stream.keys())
            assert runtime.attach_cached(items)
            response = runtime.enqueue(ids, items, 20)
            text = "".join([delta async for delta in response]).lower()
            return text, response.request

    text, req = asyncio.run(go())
    assert "red" in text
    assert all(item.data is None and item.hit for item in req.media)
