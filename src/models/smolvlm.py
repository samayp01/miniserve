import json

import mlx.core as mx
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import get_model_path, load_model, prepare_inputs
from transformers import AutoTokenizer
from transformers.models.idefics3 import Idefics3Processor
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil

from src.engine.request import Media
from src.models.base import ModelAdapter

VLM_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"


def load_vlm(name=VLM_NAME):
    path = get_model_path(name)
    model = load_model(path)
    processor = Idefics3Processor(
        image_processor=Idefics3ImageProcessorPil.from_pretrained(path),
        tokenizer=AutoTokenizer.from_pretrained(path),
        image_seq_len=json.loads((path / "processor_config.json").read_text())["image_seq_len"],
        chat_template=json.loads((path / "chat_template.json").read_text())["chat_template"],
    )
    return model, processor


class SmolVLMAdapter(ModelAdapter):
    def __init__(self, name=VLM_NAME):
        self.model, self.processor = load_vlm(name)
        self.image_token = self.model.config.image_token_index
        self.eos_token = self.processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")

    @property
    def layers(self):
        return self.model.language_model.layers

    def prepare(self, prompt, media=()):
        for item in media:
            if item["type"] != "image":
                raise NotImplementedError(f"{type(self).__name__} can't take {item['type']}")
        if len(media) > 1:
            raise NotImplementedError(f"{type(self).__name__} takes one image per request")
        images = [item["data"] for item in media]
        text = apply_chat_template(self.processor, self.model.config, prompt, num_images=len(images))
        inputs = prepare_inputs(self.processor, images=images or None, prompts=[text], image_token_index=self.image_token)
        ids = inputs["input_ids"][0].tolist()
        positions = [i for i, t in enumerate(ids) if t == self.image_token]
        return ids, [Media(positions=positions, data=inputs["pixel_values"][0])] if images else []

    def encode(self, media):
        placeholders = mx.array([[self.image_token] * len(media.positions)])
        return self.model.get_input_embeddings(placeholders, media.data[None]).inputs_embeds[0]

    def embed_tokens(self, token_ids):
        return self.model.language_model.embed_tokens(mx.array(token_ids))[None]

    def prefill(self, vectors, cache):
        return self.model.language_model(None, inputs_embeds=vectors, cache=cache).logits

    def decode(self, token_ids, cache):
        return self.model.language_model(mx.array(token_ids), cache=cache, mask=cache[0].make_mask(1)).logits
