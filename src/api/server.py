import asyncio
import base64
import io
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.api.runtime import GenerationError, MiniserveRuntime
from src.models.registry import load_adapter

runtime = None


@asynccontextmanager
async def lifespan(app):
    global runtime
    if runtime is None:
        runtime = MiniserveRuntime(load_adapter())
    async with runtime.running():
        yield


app = FastAPI(lifespan=lifespan)


class MediaItem(BaseModel):
    type: str
    data: str


class Prompt(BaseModel):
    prompt: str
    max_tokens: int = Field(128, gt=0)
    media: list[MediaItem] = []


def decode_media(item):
    raw = base64.b64decode(item.data)
    return {"type": item.type, "data": Image.open(io.BytesIO(raw)).convert("RGB") if item.type == "image" else raw}


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


async def events(stream, is_disconnected):
    try:
        async for delta in stream:
            if await is_disconnected():
                return
            yield _sse({"delta": delta})
        yield _sse({"done": True, **stream.stats})
    except GenerationError as e:
        yield _sse({"error": str(e)})
    finally:
        stream.cancel()


def _prepare(body):
    return runtime.adapter.prepare(body.prompt, [decode_media(m) for m in body.media])


@app.post("/generate")
async def generate(body: Prompt, request: Request):
    arrival = time.time()
    try:
        ids, items = await asyncio.to_thread(_prepare, body)
        stream = runtime.enqueue(ids, items, body.max_tokens, arrival)
    except (ValueError, NotImplementedError, UnidentifiedImageError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(events(stream, request.is_disconnected), media_type="text/event-stream")


@app.get("/metrics")
async def metrics():
    return runtime.metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
