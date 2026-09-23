import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.model_runner import model, tokenizer
from src.runtime import GenerationError, MiniserveRuntime

runtime = MiniserveRuntime(model, tokenizer)


@asynccontextmanager
async def lifespan(app):
    async with runtime.running():
        yield


app = FastAPI(lifespan=lifespan)


class Prompt(BaseModel):
    prompt: str
    max_tokens: int = Field(128, gt=0)


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


@app.post("/generate")
async def generate(body: Prompt, request: Request):
    try:
        stream = runtime.submit(body.prompt, body.max_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StreamingResponse(events(stream, request.is_disconnected), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
