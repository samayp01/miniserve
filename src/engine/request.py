import time
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Media:
    positions: list[int]
    embeds: mx.array


class Request:
    def __init__(self, prompt_tokens, max_output_tokens=128, media=()):
        self.prompt_tokens = prompt_tokens
        self.media = list(media)
        self.cache = None
        self.output_tokens = []
        self.done = False
        self.max_output_tokens = max_output_tokens
        self.arrival_time = time.time()
        self.admitted_time = None
        self.first_token_time = None
        self.finish_time = None
        self.prefilled = False
        self.prefill_pos = 0

    def mark_admitted(self):
        if self.admitted_time is None:
            self.admitted_time = time.time()

    def mark_done(self):
        self.done = True
        self.finish_time = time.time()

    def yield_token(self, token):
        if token is not None:
            self.output_tokens.append(token)
            if self.first_token_time is None:
                self.first_token_time = time.time()