import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from transformers import AutoTokenizer


st.set_page_config(
    page_title="Transformer LLM Playground",
    page_icon="🧠",
    layout="wide",
)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def compute_attention(tokens, d_model=8):
    np.random.seed(42)

    embeddings = np.random.rand(len(tokens), d_model)

    w_q = np.random.rand(d_model, d_model)
    w_k = np.random.rand(d_model, d_model)
    w_v = np.random.rand(d_model, d_model)

    q = embeddings @ w_q
    k = embeddings @ w_k
    v = embeddings @ w_v

    scores = q @ k.T / np.sqrt(d_model)
    attention_weights = softmax(scores)
    output = attention_weights @ v

    return scores, attention_weights, output


def main():
    st.title("🧠 Transformer LLM Playground")

    st.write(
        "Explore tokenization and toy self-attention with small interactive examples."
    )

    tab1, tab2 = st.tabs(["Tokenizer Explorer", "Attention Heatmap"])

    with tab1:
        st.header("Tokenizer Explorer")

        model_name = st.selectbox(
            "Choose a tokenizer",
            [
                "bert-base-uncased",
                "gpt2",
                "distilbert-base-uncased",
            ],
        )

        text = st.text_area(
            "Enter text",
            "Transformers are amazing because they understand context.",
        )

        if st.button("Tokenize"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)

            st.subheader("Tokens")
            st.write(tokens)

            st.subheader("Token IDs")
            st.write(token_ids)

            st.subheader("Token Count")
            st.write(len(tokens))

    with tab2:
        st.header("Toy Attention Heatmap")

        token_input = st.text_input(
            "Enter tokens separated by spaces",
            "The cat sat near bank",
        )

        tokens = token_input.split()

        if st.button("Generate Attention Heatmap"):
            if len(tokens) == 0:
                st.warning("Please enter at least one token.")
                return

            scores, attention_weights, output = compute_attention(tokens)

            fig, ax = plt.subplots(figsize=(7, 5))

            heatmap = ax.imshow(attention_weights)

            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))

            ax.set_xticklabels(tokens)
            ax.set_yticklabels(tokens)

            ax.set_xlabel("Tokens being attended to")
            ax.set_ylabel("Current token")
            ax.set_title("Toy Self-Attention Heatmap")

            fig.colorbar(heatmap, ax=ax, label="Attention weight")

            st.pyplot(fig)

            st.subheader("Attention Scores")
            st.write(scores)

            st.subheader("Attention Weights")
            st.write(attention_weights)

            st.subheader("Contextualized Output Vectors")
            st.write(output)


if __name__ == "__main__":
    main()