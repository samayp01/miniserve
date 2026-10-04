import base64
import io
import random
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw
from transformers import AutoTokenizer

MODELS = {"llama": "mlx-community/Llama-3.2-1B-Instruct-4bit", "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct"}

QUESTIONS = [
    "Name three primary colors.",
    "Explain what a hash map is in two sentences.",
    "Write a short paragraph about the ocean.",
    "List the first five prime numbers and why they're prime.",
    "Summarize how photosynthesis works for a ten-year-old.",
    "Give three tips for writing clean code.",
]

PASSAGE = [
    "The harbor town woke before dawn, when the fishing boats slipped out past the breakwater.",
    "A hash table stores values in buckets chosen by a hash of the key, trading memory for fast lookups.",
    "Glaciers carve valleys slowly, grinding bedrock into fine silt that turns meltwater a milky blue.",
    "The committee argued for hours about the budget before agreeing to postpone the vote until spring.",
    "Photosynthesis converts light, water, and carbon dioxide into sugar and releases oxygen as a byproduct.",
    "In the old library, the card catalog still sat beside the terminals, its drawers swollen with humidity.",
    "Compilers translate source code through several passes, each lowering the program closer to machine code.",
    "The river flooded twice that decade, and each time the town rebuilt the levee a little higher.",
    "Migratory birds navigate using the sun, the stars, and a sense of the planet's magnetic field.",
    "She kept a notebook of every experiment, including the failures, which taught her the most.",
    "Railways reshaped the continent, turning week-long journeys into overnight trips between cities.",
    "A good benchmark isolates one variable at a time, so that a change in the result has one explanation.",
]

INSTRUCTION = "\n\nSummarize the text above in a few sentences."

MEDIA_QUESTIONS = {
    "image": [
        "What shapes are in this image?",
        "Describe this image.",
        "What colors do you see in this image?",
    ],
}


@dataclass(frozen=True)
class Bucket:
    name: str
    weight: float
    tokens: tuple[int, int] | None = None
    media: str | None = None


SPECS = {
    "legacy": [Bucket("short", 1.0)],
    "mixed": [
        Bucket("short", 0.60),
        Bucket("medium", 0.25, (200, 450)),
        Bucket("long", 0.15, (700, 1500)),
    ],
    "image": [
        Bucket("short", 0.45),
        Bucket("medium", 0.19, (200, 450)),
        Bucket("long", 0.11, (700, 1500)),
        Bucket("image", 0.25, media="image"),
    ],
}


def _image(rng):
    image = Image.new("RGB", (512, 512), tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    for _ in range(3):
        x, y = rng.randrange(400), rng.randrange(400)
        box = (x, y, x + rng.randrange(40, 112), y + rng.randrange(40, 112))
        fill = tuple(rng.randrange(256) for _ in range(3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)(box, fill=fill)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()



GENERATORS = {"image": _image}


class Workload:
    def __init__(self, spec, seed=0, model="llama", repeat=0.0):
        self.spec = spec
        self.buckets = SPECS[spec]
        self.seed = seed
        self.repeat = repeat
        self.seen = {}
        self.tok = AutoTokenizer.from_pretrained(MODELS[model])

    def _templated_len(self, text):
        return len(self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=False))

    def _passage(self, rng, target, nonce):
        overhead = self._templated_len(nonce + INSTRUCTION.strip())
        sentences, ids = [], []
        while len(ids) < target - overhead:
            sentences.append(rng.choice(PASSAGE))
            ids = self.tok.encode(" ".join(sentences), add_special_tokens=False)
        return nonce + self.tok.decode(ids[:target - overhead], clean_up_tokenization_spaces=False) + INSTRUCTION

    def _prompt(self, rng, nonce):
        if len(self.buckets) == 1:
            bucket = self.buckets[0]
        else:
            bucket = rng.choices(self.buckets, weights=[b.weight for b in self.buckets])[0]
        if bucket.media:
            text = nonce + rng.choice(MEDIA_QUESTIONS[bucket.media])
            seen = self.seen.setdefault(bucket.media, [])
            pick = random.Random(f"{self.seed}{nonce}")
            if self.repeat and seen and pick.random() < self.repeat:
                data = pick.choice(seen)
            else:
                data = GENERATORS[bucket.media](pick)
                seen.append(data)
            media = [{"type": bucket.media, "data": data}]
            return {"bucket": bucket.name, "prompt": text, "prompt_tokens": self._templated_len(text), "media": media}
        if bucket.tokens is None:
            text = nonce + rng.choice(QUESTIONS)
        else:
            text = self._passage(rng, rng.randint(*bucket.tokens), nonce)
        return {"bucket": bucket.name, "prompt": text, "prompt_tokens": self._templated_len(text)}

    def schedule(self, qps, num_requests, tag):
        rng = random.Random(self.seed)
        self.seen = {}
        reqs = []
        for i in range(num_requests):
            req = self._prompt(rng, f"[{tag}-{i}] ")
            req["gap"] = rng.expovariate(qps) if i < num_requests - 1 else 0.0
            reqs.append(req)
        return reqs

    def describe(self):
        return {"spec": self.spec, "seed": self.seed, "buckets": [asdict(b) for b in self.buckets]}
