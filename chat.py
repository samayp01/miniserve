import sys
from engine import generate

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chat.py <prompt>")
        sys.exit(1)

    prompt = sys.argv[1]
    text, _ = generate(prompt, stop_on_eos=True)
    print(text)
