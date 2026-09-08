import sys
from src.model_runner import model, tokenizer
from src.engine import Engine
from src.request import Request
from src.cache.paged_cache import make_block_pools

def stream(engine, prompt):
    ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    req = Request(ids)
    engine.add_request(req)

    printed = ""
    while not req.done:
        engine.step()
        text = tokenizer.decode(req.output_tokens, clean_up_tokenization_spaces=False)
        sys.stdout.write(text[len(printed):])
        sys.stdout.flush()
        printed = text
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.chat <prompt> [<prompt> ...]")
        sys.exit(1)

    pools = make_block_pools(model, num_blocks=4096, block_size=16)
    engine = Engine(pools)
    for prompt in sys.argv[1:]:
        stream(engine, prompt)
