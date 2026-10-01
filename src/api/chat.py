import asyncio
import sys

from src.api.runtime import MiniserveRuntime
from src.models.registry import load_adapter


async def main(prompts):
    runtime = MiniserveRuntime(load_adapter())
    async with runtime.running():
        for prompt in prompts:
            async for delta in runtime.submit(prompt):
                sys.stdout.write(delta)
                sys.stdout.flush()
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.api.chat <prompt> [<prompt> ...]")
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
