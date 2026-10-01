import os

from src.models.llama import LlamaAdapter
from src.models.smolvlm import SmolVLMAdapter

ADAPTERS = {"llama": LlamaAdapter, "smolvlm": SmolVLMAdapter}


def load_adapter(name=None):
    name = name or os.environ.get("MINISERVE_MODEL", "llama")
    if name not in ADAPTERS:
        raise ValueError(f"unknown model {name!r}, choose from {', '.join(ADAPTERS)}")
    return ADAPTERS[name]()
