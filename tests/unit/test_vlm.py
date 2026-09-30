import mlx.core as mx
from PIL import Image, ImageDraw
from mlx_vlm.models.cache import KVCache
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

from src.vlm import load_vlm

model, processor = load_vlm()
IMAGE_TOKEN = model.config.image_token_index
QUESTION = "What shapes are in this image, and what colors are they?"


def _shapes_image():
    image = Image.new("RGB", (512, 512), (30, 60, 160))
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 50, 250, 300), fill=(220, 30, 30))
    draw.rectangle((280, 260, 460, 460), fill=(250, 220, 40))
    return image


def _inputs(image):
    prompt = apply_chat_template(processor, model.config, QUESTION, num_images=1)
    return prepare_inputs(processor, images=[image], prompts=[prompt], image_token_index=IMAGE_TOKEN)


def _answer(image, max_tokens=30):
    inputs = _inputs(image)
    extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
    cache = [KVCache() for _ in model.language_model.layers]
    logits = model(inputs["input_ids"], inputs["pixel_values"], cache=cache, **extra).logits
    eos = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
    tokens = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        if token.item() == eos:
            break
        tokens.append(token.item())
        logits = model.language_model(token[None], cache=cache).logits
    return processor.tokenizer.decode(tokens)


def test_prompt_reserves_one_slot_per_image_vector():
    inputs = _inputs(_shapes_image())
    views = inputs["pixel_values"].shape[1]
    slots = int((inputs["input_ids"] == IMAGE_TOKEN).sum().item())
    assert views == 17
    assert slots == 64 * views


def test_describes_image_contents():
    answer = _answer(_shapes_image()).lower()
    assert "red" in answer and "circle" in answer
    assert "yellow" in answer and "square" in answer
