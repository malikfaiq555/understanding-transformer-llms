import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(q, k, v):
    d_k = q.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = attention_weights @ v
    return scores, attention_weights, output


def main():
    np.random.seed(42)

    tokens = ["I", "love", "transformers"]
    d_model = 8

    embeddings = np.random.rand(len(tokens), d_model)

    w_q = np.random.rand(d_model, d_model)
    w_k = np.random.rand(d_model, d_model)
    w_v = np.random.rand(d_model, d_model)

    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v

    scores, attention_weights, output = scaled_dot_product_attention(q, k, v)

    print("Tokens:")
    print(tokens)

    print("\nAttention Scores:")
    print(scores)

    print("\nAttention Weights:")
    print(attention_weights)

    print("\nContextualized Output Vectors:")
    print(output)


if __name__ == "__main__":
    main()