from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

def main():
    print(generate(model, tokenizer, prompt="Hi", max_tokens=50))


if __name__ == "__main__":
    main()
