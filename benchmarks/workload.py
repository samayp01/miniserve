import random
from dataclasses import asdict, dataclass

from transformers import AutoTokenizer

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

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


@dataclass(frozen=True)
class Bucket:
    name: str
    weight: float
    tokens: tuple[int, int] | None = None


SPECS = {
    "legacy": [Bucket("short", 1.0)],
    "mixed": [
        Bucket("short", 0.60),
        Bucket("medium", 0.25, (200, 450)),
        Bucket("long", 0.15, (700, 1500)),
    ],
}


class Workload:
    def __init__(self, spec, seed=0):
        self.spec = spec
        self.buckets = SPECS[spec]
        self.seed = seed
        self.tok = AutoTokenizer.from_pretrained(MODEL)

    def _templated_len(self, text):
        return len(self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=False))

    def _passage(self, rng, target):
        overhead = self._templated_len(INSTRUCTION.strip())
        sentences, ids = [], []
        while len(ids) < target - overhead:
            sentences.append(rng.choice(PASSAGE))
            ids = self.tok.encode(" ".join(sentences), add_special_tokens=False)
        return self.tok.decode(ids[:target - overhead], clean_up_tokenization_spaces=False) + INSTRUCTION

    def _prompt(self, rng):
        if len(self.buckets) == 1:
            bucket = self.buckets[0]
        else:
            bucket = rng.choices(self.buckets, weights=[b.weight for b in self.buckets])[0]
        if bucket.tokens is None:
            text = rng.choice(QUESTIONS)
        else:
            text = self._passage(rng, rng.randint(*bucket.tokens))
        return {"bucket": bucket.name, "prompt": text, "prompt_tokens": self._templated_len(text)}

    def schedule(self, qps, num_requests):
        """Requests with the gap (seconds) to wait after sending each; identical for a given seed and qps."""
        rng = random.Random(self.seed)
        reqs = []
        for i in range(num_requests):
            req = self._prompt(rng)
            req["gap"] = rng.expovariate(qps) if i < num_requests - 1 else 0.0
            reqs.append(req)
        return reqs

    def describe(self):
        return {"spec": self.spec, "seed": self.seed, "buckets": [asdict(b) for b in self.buckets]}
