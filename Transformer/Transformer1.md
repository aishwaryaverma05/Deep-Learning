# Transformer — Quick Revision Notes

## 1. What is a Transformer?

### Definition

A **Transformer** is a deep learning architecture designed to process sequential data, especially text, using a mechanism called **Self-Attention**.

It was introduced in the research paper *Attention Is All You Need* (2017).

Unlike RNNs and LSTMs, Transformers can process multiple tokens in parallel during training and learn relationships between words that are far apart in a sentence.

### Why do we need Transformers?

Before Transformers, NLP mainly used RNNs and LSTMs.

* **RNN:** Processes tokens one by one, making parallel processing difficult.
* **LSTM:** Learns long-term dependencies better than a basic RNN but still processes tokens sequentially.
* **Transformer:** Uses self-attention to learn relationships between tokens and enables parallel processing during training.

### Example

Sentence: "The animal didn't cross the road because it was tired."

To understand this sentence, the model needs to identify what the word **"it"** refers to.

Self-attention helps the model learn relationships between "it" and other relevant words in the sentence.

## 2. How Does a Transformer Work?

1. **Input Text:** Example — `"I love machine learning"`
2. **Tokenization:** Splits text into tokens.
3. **Embedding:** Converts tokens into numerical vectors.
4. **Positional Encoding:** Adds information about token positions.
5. **Transformer Blocks:** Use self-attention and feed-forward networks to learn relationships between tokens.
6. **Output Layer:** Produces predictions for a task, such as sentiment classification, translation or next-token prediction.

**Note:** The exact architecture depends on the task and model type.

## 3. Key Features of Transformers

1. **Self-Attention:** Determines which other tokens are relevant to a token.
2. **Parallel Processing:** Processes multiple tokens simultaneously during training.
3. **Long-Range Dependencies:** Learns relationships between distant tokens.
4. **Contextual Representations:** Represents a word according to its surrounding context.
5. **Scalability:** Can be trained on large datasets to build powerful models.

## 4. Applications of Transformers

* **Large Language Models:** GPT and similar models.
* **Text Generation:** Generating answers, stories and code.
* **Machine Translation:** Translating between languages.
* **Text Understanding:** Sentiment analysis, classification and question answering.
* **Multimodal AI:** Processing text, images, audio and other data.

## 5. Important Points to Remember

* Transformer is an architecture, not one specific model.
* Self-attention is its key innovation.
* It was introduced in the 2017 paper *Attention Is All You Need*.
* It uses token embeddings and positional information.
* The original Transformer contains an encoder and a decoder.
* Modern Transformers may use an encoder-only, decoder-only or encoder-decoder architecture.

## 6. One-Minute Revision

* **Transformer:** Attention-based deep learning architecture.
* **Self-Attention:** Learns relationships between tokens.
* **Tokenization:** Text to tokens.
* **Embedding:** Tokens to numerical vectors.
* **Positional Encoding:** Provides token position information.
* **Parallel Processing:** Processes multiple tokens simultaneously during training.
* **Encoder:** Builds contextual representations of input.
* **Decoder:** Generates outputs using the architecture's available context.

**In one line:** A Transformer uses attention mechanisms to learn relationships between tokens efficiently and forms the foundation of many modern NLP and generative AI systems.



# 2. What is Self-Attention?

## Definition

Self-Attention is a mechanism in a Transformer that helps each token understand how much attention it should pay to other tokens in the same input sequence to understand the context and meaning.

In simple words, it helps the model identify which words are important for understanding a particular word.

## Why Do We Need Self-Attention?

The meaning of a word often depends on the other words present in a sentence.

**Example:**

> "The animal didn't cross the road because it was tired."

To understand the word **"it"**, the model needs to identify which other word it is referring to. Self-attention helps the model learn the relationship between **"it"** and **"animal"** by considering the surrounding context.

## How Does Self-Attention Work?

Consider the sentence:

> "The cat sat on the mat."

Self-attention works through the following steps:

1. **Token Embeddings:** Each token is converted into a numerical representation called an embedding.

2. **Query, Key, and Value:** For each token, the model generates three vectors — Query (Q), Key (K), and Value (V).

3. **Calculate Attention Scores:** The Query of one token is compared with the Keys of other tokens to determine their relevance.

4. **Apply Softmax:** The attention scores are converted into normalized weights using the Softmax function.

5. **Calculate Weighted Values:** The model combines the Value vectors using these attention weights to create a context-aware representation of each token.

## Understanding Query, Key, and Value (Q, K, V)

| Component     | Meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| **Query (Q)** | What information is this token looking for?                |
| **Key (K)**   | What information does this token offer for matching?       |
| **Value (V)** | What information does this token contribute to the output? |

Think of it like a search system:

* **Query:** What am I looking for?
* **Key:** How well does this item match my search?
* **Value:** What information should I retrieve from this item?

## Important Points to Remember

* Self-attention allows each token to consider other tokens in the same sequence.
* It helps the model learn contextual relationships between words.
* Each token gets its own Query, Key, and Value vectors.
* Attention weights determine how much information other tokens contribute to a token's representation.
* Self-attention is a core component of Transformer architectures.

## Quick Revision

**In one line:** Self-attention helps each token gather relevant information from other tokens to build a better understanding of its meaning in context.

**Remember:** Q = Query, K = Key, V = Value.

<img width="1312" height="1199" alt="image" src="https://github.com/user-attachments/assets/ddbcae1e-e599-4da5-ba48-89ab067b176e" />


