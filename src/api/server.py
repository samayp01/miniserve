import asyncio
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.api.runtime import GenerationError, MiniserveRuntime
from src.engine.scheduler import FifoScheduler, PriorityScheduler
from src.models.registry import load_adapter

runtime = None


def load_scheduler():
    weight = os.environ.get("MINISERVE_AGE_WEIGHT")
    return PriorityScheduler(float(weight)) if weight is not None else FifoScheduler()


def engine_settings():
    settings = {"scheduler": load_scheduler()}
    for name, key in (("MINISERVE_MAX_BATCH", "max_batch"), ("MINISERVE_KV_BLOCKS", "num_blocks"),
                      ("MINISERVE_SWAP_MIN_TOKENS", "swap_min_tokens")):
        if name in os.environ:
            settings[key] = int(os.environ[name])
    return settings


@asynccontextmanager
async def lifespan(app):
    global runtime
    if runtime is None:
        runtime = MiniserveRuntime(load_adapter(), **engine_settings())
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


class StreamOptions(BaseModel):
    fps: float = Field(1.0, gt=0)
    window: int = Field(30, gt=0)


class Frame(BaseModel):
    data: str


class Question(BaseModel):
    prompt: str
    max_tokens: int = Field(128, gt=0)


def _decode_image(raw):
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"invalid image: {e}") from e


DECODERS = {"image": _decode_image, "frame": _decode_image}


def decode_media(item):
    raw = base64.b64decode(item.data)
    decode = DECODERS.get(item.type)
    return {"type": item.type, "data": decode(raw) if decode else raw}


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


async def _respond(prompt, max_tokens, load_media, request):
    arrival = time.time()
    try:
        ids, items = await asyncio.to_thread(lambda: runtime.adapter.prepare(prompt, load_media()))
        stream = runtime.enqueue(ids, items, max_tokens, arrival)
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(events(stream, request.is_disconnected), media_type="text/event-stream")


def _media_stream(stream_id):
    stream = runtime.media_streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"unknown stream {stream_id}")
    return stream


def _prepare_frame(data):
    frame = decode_media(MediaItem(type="frame", data=data))
    return frame["data"], runtime.adapter.prepare("", [frame])[1]


@app.post("/generate")
async def generate(body: Prompt, request: Request):
    return await _respond(body.prompt, body.max_tokens, lambda: [decode_media(m) for m in body.media], request)


@app.post("/streams")
async def open_stream(body: StreamOptions):
    return {"stream_id": runtime.open_stream(body.fps, body.window).id}


@app.post("/streams/{stream_id}/frames")
async def add_frame(stream_id: str, body: Frame):
    stream = _media_stream(stream_id)
    reason = stream.admit(time.time())
    if reason:
        return {"kept": False, "reason": reason}
    try:
        frame, items = await asyncio.to_thread(_prepare_frame, body.data)
    except (ValueError, NotImplementedError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    runtime.ingest(stream, frame, items)
    return {"kept": True}


@app.post("/streams/{stream_id}/ask")
async def ask(stream_id: str, body: Question, request: Request):
    stream = _media_stream(stream_id)
    if not stream.frames:
        raise HTTPException(status_code=400, detail="stream has no frames yet")
    media = stream.media()
    return await _respond(body.prompt, body.max_tokens, lambda: media, request)


@app.delete("/streams/{stream_id}")
async def close_stream(stream_id: str):
    stream = runtime.close_stream(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail=f"unknown stream {stream_id}")
    return {"kept": stream.kept, "skipped": stream.skipped, "dropped": stream.dropped}


@app.get("/metrics")
async def metrics():
    return runtime.metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
