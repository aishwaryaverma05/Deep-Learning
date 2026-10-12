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
