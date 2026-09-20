import asyncio
from contextlib import asynccontextmanager

from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools


class StreamDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.tokens = []
        self.prefix_offset = 0
        self.read_offset = 0

    def _window(self):
        prefix = self.tok.decode(self.tokens[self.prefix_offset:self.read_offset], clean_up_tokenization_spaces=False)
        whole = self.tok.decode(self.tokens[self.prefix_offset:], clean_up_tokenization_spaces=False)
        return prefix, whole

    def add(self, token):
        self.tokens.append(token)
        prefix, whole = self._window()
        if len(whole) > len(prefix) and not whole.endswith("�"):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            return whole[len(prefix):]
        return ""

    def flush(self):
        prefix, whole = self._window()
        self.prefix_offset = self.read_offset
        self.read_offset = len(self.tokens)
        return whole[len(prefix):]


def _stats(req):
    return {
        "tokens": len(req.output_tokens),
        "ttft_ms": round((req.first_token_time - req.arrival_time) * 1000, 1) if req.first_token_time else None,
        "latency_ms": round((req.finish_time - req.arrival_time) * 1000, 1) if req.finish_time else None,
    }


class Stream:
    def __init__(self, request, tokenizer):
        self.request = request
        self.stats = None
        self._detok = StreamDetokenizer(tokenizer)
        self._queue = asyncio.Queue()
        self._cursor = 0

    def advance(self):
        req = self.request
        delta = ""
        while self._cursor < len(req.output_tokens):
            delta += self._detok.add(req.output_tokens[self._cursor])
            self._cursor += 1
        if req.done:
            delta += self._detok.flush()
        if delta:
            self._queue.put_nowait(delta)
        if req.done:
            self.stats = _stats(req)
            self._queue.put_nowait(None)
        return req.done

    def __aiter__(self):
        return self

    async def __anext__(self):
        delta = await self._queue.get()
        if delta is None:
            raise StopAsyncIteration
        return delta


class MiniserveRuntime:
    def __init__(self, model, tokenizer, num_blocks=4096, block_size=16, **engine_kwargs):
        self.tokenizer = tokenizer
        self.engine = Engine(make_block_pools(model, num_blocks, block_size), **engine_kwargs)
        self.streams = []

    def submit(self, prompt, max_tokens=128):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        req = Request(ids, max_output_tokens=max_tokens)
        self.engine.add_request(req)
        stream = Stream(req, self.tokenizer)
        self.streams.append(stream)
        return stream

    def _drain(self):
        self.streams = [s for s in self.streams if not s.advance()]

    async def run(self):
        while True:
            if self.engine.running or self.engine.waiting:
                self.engine.step()
                self._drain()
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.005)

    @asynccontextmanager
    async def running(self):
        task = asyncio.create_task(self.run())
        try:
            yield self
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
