from src.request import Request

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
