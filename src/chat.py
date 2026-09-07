import sys
from src.model_runner import model, tokenizer
from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.chat <prompt1> <prompt2> ...")
        sys.exit(1)

    prompts = sys.argv[1:]
    pools = make_block_pools(model, num_blocks=4096, block_size=16)
    engine = Engine(pools)

    reqs = []
    for p in prompts:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        r = Request(ids)
        reqs.append(r)
        engine.add_request(r)

    engine.run()

    for r in reqs:
        print(tokenizer.decode(r.output_tokens, clean_up_tokenization_spaces=False))
