import asyncio
import sys

from src.model_runner import model, tokenizer
from src.runtime import MiniserveRuntime


async def main(prompts):
    runtime = MiniserveRuntime(model, tokenizer)
    async with runtime.running():
        for prompt in prompts:
            async for delta in runtime.submit(prompt):
                sys.stdout.write(delta)
                sys.stdout.flush()
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.chat <prompt> [<prompt> ...]")
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
