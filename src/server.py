import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.model_runner import model, tokenizer
from src.runtime import MiniserveRuntime

runtime = MiniserveRuntime(model, tokenizer)


@asynccontextmanager
async def lifespan(app):
    async with runtime.running():
        yield


app = FastAPI(lifespan=lifespan)


class Prompt(BaseModel):
    prompt: str
    max_tokens: int = 128


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/generate")
async def generate(body: Prompt):
    try:
        stream = runtime.submit(body.prompt, body.max_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async def events():
        async for delta in stream:
            yield _sse({"delta": delta})
        yield _sse({"done": True, **stream.stats})

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
