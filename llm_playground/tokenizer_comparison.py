from transformers import AutoTokenizer


def compare_tokenizers(text):
    model_names = [
        "bert-base-uncased",
        "gpt2",
        "distilbert-base-uncased",
    ]

    for model_name in model_names:
        print("=" * 80)
        print(f"Tokenizer: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)

        print("\nInput text:")
        print(text)

        print("\nTokens:")
        print(tokens)

        print("\nToken IDs:")
        print(token_ids)

        print("\nToken count:")
        print(len(tokens))
        print()


def main():
    text = "Transformers are amazing because they understand context."
    compare_tokenizers(text)


if __name__ == "__main__":
    main()