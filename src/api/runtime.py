import asyncio
import logging
from contextlib import asynccontextmanager

import mlx.core as mx

from src.engine.engine import Engine
from src.engine.request import Request
from src.cache.paged_cache import make_block_pools

log = logging.getLogger("miniserve")


class GenerationError(RuntimeError):
    pass


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


def _ms(start, end):
    return round((end - start) * 1000, 1) if start and end else None


def _stats(req):
    return {
        "tokens": len(req.output_tokens),
        "prompt_tokens": len(req.prompt_tokens),
        "media_hits": sum(item.hit for item in req.media),
        "ttft_ms": _ms(req.arrival_time, req.first_token_time),
        "latency_ms": _ms(req.arrival_time, req.finish_time),
        "prepare_ms": _ms(req.arrival_time, req.prepared_time),
        "queue_ms": _ms(req.prepared_time, req.admitted_time),
        "encode_ms": _ms(req.admitted_time, req.encoded_time),
        "prefill_ms": _ms(req.encoded_time, req.first_token_time),
        "decode_ms": _ms(req.first_token_time, req.finish_time),
    }


class Stream:
    def __init__(self, request, tokenizer, on_cancel):
        self.request = request
        self.stats = None
        self.error = None
        self._detok = StreamDetokenizer(tokenizer)
        self._queue = asyncio.Queue()
        self._cursor = 0
        self._on_cancel = on_cancel

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

    def fail(self, error):
        self.error = error
        self._queue.put_nowait(None)

    def cancel(self):
        self._on_cancel(self)

    def __aiter__(self):
        return self

    async def __anext__(self):
        delta = await self._queue.get()
        if delta is None:
            if self.error is not None:
                raise GenerationError(str(self.error)) from self.error
            raise StopAsyncIteration
        return delta


class MiniserveRuntime:
    def __init__(self, adapter, num_blocks=4096, block_size=16, **engine_kwargs):
        self.adapter = adapter
        self.engine = Engine(make_block_pools(adapter, num_blocks, block_size), adapter, **engine_kwargs)
        self.streams = []

    def submit(self, prompt, max_tokens=128, media=()):
        return self.enqueue(*self.adapter.prepare(prompt, media), max_tokens)

    def enqueue(self, ids, items, max_tokens=128, arrival_time=None):
        req = Request(ids, max_output_tokens=max_tokens, media=items, arrival_time=arrival_time)
        self.engine.add_request(req)
        stream = Stream(req, self.adapter.tokenizer, self.cancel)
        self.streams.append(stream)
        return stream

    def metrics(self):
        return {
            "waiting": len(self.engine.waiting),
            "running": len(self.engine.running),
            "preemptions": self.engine.preemptions,
            "free_blocks": len(self.engine.pools[0].allocator.free),
            "total_blocks": self.engine.capacity,
            "active_mb": round(mx.get_active_memory() / 2**20, 1),
            "peak_mb": round(mx.get_peak_memory() / 2**20, 1),
            "scheduler": self.engine.scheduler.name,
            "max_batch": self.engine.max_batch,
            "swap_min_tokens": self.engine.swap_min_tokens,
            "encode_cache_entries": len(self.engine.encode_cache.entries),
            "encode_cache_mb": round(self.engine.encode_cache.bytes / 2**20, 1),
            "swaps": self.engine.swaps,
            "swap_mb": round(self.engine.swap.bytes / 2**20, 1) if self.engine.swap else 0.0,
        }

    def cancel(self, stream):
        if stream in self.streams:
            self.streams.remove(stream)
            self.engine.abort(stream.request)

    def _settle(self, stream):
        try:
            return stream.advance()
        except Exception as error:
            log.exception("streaming request output failed")
            self.engine.abort(stream.request)
            stream.fail(error)
            return True

    def _drain(self):
        self.streams = [s for s in self.streams if not self._settle(s)]

    def _fail_running(self, error):
        failed = list(self.engine.running)
        for req in failed:
            self.engine.abort(req)
        failed_ids = {id(req) for req in failed}
        survivors = []
        for stream in self.streams:
            if id(stream.request) not in failed_ids:
                survivors.append(stream)
            elif not self._settle(stream):
                stream.fail(error)
        self.streams = survivors

    async def run(self):
        while True:
            if self.engine.running or self.engine.waiting:
                try:
                    self.engine.step()
                except Exception as error:
                    log.exception("engine step failed")
                    self._fail_running(error)
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
