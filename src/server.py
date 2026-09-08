import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.model_runner import model, tokenizer
from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools

pools = make_block_pools(model, num_blocks=4096, block_size=16)
engine = Engine(pools)

async def _engine_loop():
    while True:
        if engine.running or engine.waiting:
            engine.step()
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(0.005)

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(_engine_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)


class StreamDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.tokens = []
        self.prefix_offset = 0
        self.read_offset = 0

    def add(self, token):
        self.tokens.append(token)
        prefix = self.tok.decode(self.tokens[self.prefix_offset:self.read_offset], clean_up_tokenization_spaces=False)
        whole = self.tok.decode(self.tokens[self.prefix_offset:], clean_up_tokenization_spaces=False)
        if len(whole) > len(prefix) and not whole.endswith("�"):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.tokens)
            return whole[len(prefix):]
        return ""


class Prompt(BaseModel):
    prompt: str
    max_tokens: int = 128


@app.post("/generate")
async def generate(body: Prompt):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": body.prompt}], add_generation_prompt=True)
    req = Request(ids, max_output_tokens=body.max_tokens)
    engine.add_request(req)

    async def events():
        detok = StreamDetokenizer(tokenizer)
        i = 0
        emitted = 0
        while True:
            toks = list(req.output_tokens)
            delta = ""
            while i < len(toks):
                delta += detok.add(toks[i])
                i += 1
            if delta:
                emitted += len(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            if req.done:
                break
            await asyncio.sleep(0.01)

        full = tokenizer.decode(list(req.output_tokens), clean_up_tokenization_spaces=False)
        if len(full) > emitted:
            yield f"data: {json.dumps({'delta': full[emitted:]})}\n\n"

        meta = {
            "done": True,
            "tokens": len(req.output_tokens),
            "ttft_ms": round((req.first_token_time - req.arrival_time) * 1000, 1) if req.first_token_time else None,
            "latency_ms": round((req.finish_time - req.arrival_time) * 1000, 1) if req.finish_time else None,
        }
        yield f"data: {json.dumps(meta)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
