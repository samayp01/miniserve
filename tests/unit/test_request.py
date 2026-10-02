import mlx.core as mx

from src.engine.model_runner import embed
from src.engine.request import Media, Request
from tests.unit.fakes import FakeAdapter

FAKE = FakeAdapter()

def test_defaults():
    req = Request([1, 2, 3])
    assert req.prompt_tokens == [1, 2, 3]
    assert req.output_tokens == []
    assert req.done is False
    assert req.prefilled is False
    assert req.first_token_time is None
    assert req.finish_time is None

def test_yield_token_appends():
    req = Request([1])
    req.yield_token(42)
    req.yield_token(43)
    assert req.output_tokens == [42, 43]

def test_first_token_time_set_once():
    req = Request([1])
    req.yield_token(42)
    first = req.first_token_time
    assert first is not None
    req.yield_token(43)
    assert req.first_token_time == first

def test_yield_token_ignores_none():
    req = Request([1])
    req.yield_token(None)
    assert req.output_tokens == []
    assert req.first_token_time is None

def test_mark_done():
    req = Request([1])
    req.mark_done()
    assert req.done is True
    assert req.finish_time is not None

def _image_request():
    image = Media(positions=[3, 4, 5, 6], embeds=mx.array([[10.0], [11.0], [12.0], [13.0]]))
    return Request(list(range(10)), media=[image])

def test_embed_places_media_rows_at_their_positions():
    assert embed(_image_request(), 0, 10, FAKE)[0, :, 0].tolist() == [0, 0, 0, 10, 11, 12, 13, 0, 0, 0]

def test_embed_splits_media_across_chunks():
    req = _image_request()
    assert embed(req, 0, 5, FAKE)[0, :, 0].tolist() == [0, 0, 0, 10, 11]
    assert embed(req, 5, 10, FAKE)[0, :, 0].tolist() == [12, 13, 0, 0, 0]

def test_embed_handles_several_media_items():
    a = Media(positions=[1, 2], embeds=mx.array([[5.0], [6.0]]))
    b = Media(positions=[7], embeds=mx.array([[9.0]]))
    req = Request(list(range(10)), media=[a, b])
    assert embed(req, 0, 10, FAKE)[0, :, 0].tolist() == [0, 5, 6, 0, 0, 0, 0, 9, 0, 0]

def test_text_request_has_no_media():
    assert Request([1, 2, 3]).media == []

def test_arrival_time_can_be_set_before_preparation():
    req = Request([1], arrival_time=5.0)
    assert req.arrival_time == 5.0
    assert req.prepared_time > req.arrival_time

def test_arrival_defaults_to_preparation_time():
    req = Request([1])
    assert req.arrival_time == req.prepared_time
