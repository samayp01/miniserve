import sys
from engine import generate

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chat.py <prompt1> <prompt2> ...")
        sys.exit(1)

    prompts = sys.argv[1:]
    print("\n".join(generate(prompts, stop_on_eos=True)))

