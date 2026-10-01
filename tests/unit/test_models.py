import pytest

from src.engine.model_runner import llama, tokenizer
from src.engine.request import Media


def test_llama_prepare_applies_chat_template():
    ids, media = llama.prepare("Hi")
    assert ids == tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert media == []


def test_llama_prepare_rejects_media():
    with pytest.raises(NotImplementedError, match="can't take media"):
        llama.prepare("Hi", [{"type": "image", "data": None}])


def test_encode_defaults_to_cannot_take_media():
    with pytest.raises(NotImplementedError, match="can't take media"):
        llama.encode(Media(positions=[0]), 0, 1)
