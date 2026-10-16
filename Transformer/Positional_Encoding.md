# Positional Encoding in Deep Learning

## 1. What is Positional Encoding?

Positional Encoding is a technique used in Transformers to give information about the **position of each token** in a sequence.

Self-Attention understands relationships between tokens, but by itself, it does not naturally know the order of tokens.

**Example:**

- "Dog bites man"
- "Man bites dog"

The words are the same, but their order changes the meaning. Positional Encoding helps the Transformer understand this order.

## 2. Why Do We Need Positional Encoding?

- To provide information about token positions.
- To help the model understand word order.
- To distinguish between tokens appearing at different positions.
- To help understand the context of a sequence.

## 3. How Does Positional Encoding Work?

1. The input sentence is converted into token embeddings.
2. Positional information is generated for each token.
3. The positional information is added to the token embeddings.
4. These combined representations are passed to the Transformer.

**Formula:**

`Input = Token Embedding + Positional Encoding`

## 4. Types of Positional Encoding

### A. Sinusoidal Positional Encoding

- Uses sine and cosine functions to generate position values.
- These values are calculated using fixed mathematical formulas.
- It does not require learning positional values during training.

**Formula:**

For position `pos` and dimension `i`:

`PE(pos, 2i) = sin(pos / 10000^(2i/d))`

`PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`

Here, `d` is the embedding dimension.

**Example:** The original Transformer paper used sinusoidal positional encoding.

### B. Learned Positional Encoding

- The model learns positional representations during training.
- The positional values are trainable parameters.
- The model learns useful position information from the training data.

## 5. Benefits of Positional Encoding

- Helps Transformers understand token order.
- Provides information about token positions.
- Helps the model process sequences more meaningfully.
- Makes position information available to self-attention.

## 6. Important Interview Questions

**Q1. Why is positional encoding needed in Transformers?**

It provides token-position information because self-attention alone does not inherently represent the sequence order.

**Q2. Where is positional encoding added?**

In the original Transformer, it is added to token embeddings before they enter the encoder or decoder stack.

**Q3. What are the main types of positional encoding?**

Sinusoidal positional encoding and learned positional encoding.

**Q4. Does sinusoidal positional encoding have trainable parameters?**

No. Its values are calculated using fixed sine and cosine functions.

**Q5. What is the difference between token embedding and positional encoding?**

- **Token Embedding:** Represents the token's information or meaning.
- **Positional Encoding:** Represents information about the token's position.

## Quick Revision

- **Positional Encoding:** Adds position information to token representations.
- **Why?** To help the Transformer understand token order.
- **Sinusoidal:** Uses fixed sine and cosine functions.
- **Learned:** Position representations are learned during training.
- **Formula:** Input = Token Embedding + Positional Encoding.
