import time

class Request:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens
        self.cache = None
        self.output_tokens = []
        self.done = False
        self.arrival_time = time.time()
        self.first_token_time = None
        self.prefilled = False

    def mark_done(self):
        self.done = True

    def yield_token(self, token):
        if token is not None:
            self.output_tokens.append(token)
            if self.first_token_time is None:
                self.first_token_time = time.time()