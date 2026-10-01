import asyncio

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
