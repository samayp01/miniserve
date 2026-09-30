import json

from mlx_vlm.utils import get_model_path, load_model
from transformers import AutoTokenizer
from transformers.models.idefics3 import Idefics3Processor
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil

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
