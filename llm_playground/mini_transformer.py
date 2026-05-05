import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.embed_size = embed_size

        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_size ** 0.5)

        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)

        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, forward_expansion=4):
        super().__init__()

        self.attention = SelfAttention(embed_size)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def forward(self, x):
        attention_output, attention_weights = self.attention(x)

        x = self.norm1(x + attention_output)

        feedforward_output = self.feed_forward(x)
        output = self.norm2(x + feedforward_output)

        return output, attention_weights


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_block = TransformerBlock(embed_size)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        hidden_states, attention_weights = self.transformer_block(embeddings)

        logits = self.output_layer(hidden_states)

        return logits, attention_weights


def generate(model, input_ids, max_new_tokens=5):
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(input_ids)

            last_token_logits = logits[:, -1, :]
            probabilities = torch.softmax(last_token_logits, dim=-1)

            next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def main():
    vocab = {
        "I": 0,
        "love": 1,
        "transformers": 2,
        "because": 3,
        "they": 4,
        "work": 5,
        "<pad>": 6,
    }

    inverse_vocab = {token_id: token for token, token_id in vocab.items()}

    sentence = ["I", "love"]
    input_ids = torch.tensor([[vocab[token] for token in sentence]])

    model = MiniTransformer(vocab_size=len(vocab), embed_size=16)

    logits, attention_weights = model(input_ids)

    print("Input IDs:")
    print(input_ids)

    print("\nLogits shape:")
    print(logits.shape)

    print("\nAttention weights:")
    print(attention_weights)

    generated_ids = generate(model, input_ids, max_new_tokens=5)

    generated_tokens = [
        inverse_vocab[token_id.item()]
        for token_id in generated_ids[0]
    ]

    print("\nGenerated token IDs:")
    print(generated_ids)

    print("\nGenerated tokens:")
    print(generated_tokens)


if __name__ == "__main__":
    main()