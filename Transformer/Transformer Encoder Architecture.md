# Transformer Encoder Architecture

## 1. What is a Transformer Encoder?

A Transformer Encoder is a part of the Transformer architecture that processes input tokens and creates **context-aware representations**.

It helps the model understand the meaning of each word based on the other words in the sentence.

**Example:**

In "I deposited money in the bank", the word *bank* refers to a financial institution. The encoder uses surrounding words to understand this context.

## 2. Overall Transformer Architecture

The original Transformer consists of two main parts:

- **Encoder:** Processes the input sequence and creates meaningful representations.
- **Decoder:** Uses encoder representations (along with previously generated tokens) to generate the output sequence.

The original Transformer paper uses 6 encoder blocks and 6 decoder blocks.

## 3. Structure of an Encoder Block

Each encoder block contains two main components:

1. Multi-Head Self-Attention
2. Feed-Forward Neural Network

Both components are followed by a residual connection and Layer Normalization in the original architecture.

**Flow:**

`Input → Multi-Head Attention → Add & Norm → Feed-Forward Network → Add & Norm → Output`

The output of one encoder block becomes the input to the next block.

## 4. Input Processing

Before entering the encoder, the input passes through three steps:

### A. Tokenization

The sentence is divided into tokens.

**Example:** `"How are you"` → `["How", "are", "you"]`

### B. Embedding

Each token is converted into a numerical vector because neural networks process numbers, not raw text.

In the original Transformer example, each token embedding has 512 dimensions.

### C. Positional Encoding

Positional encoding provides information about the position of each token.

It is added to the token embedding.

**Formula:**

`Input = Token Embedding + Positional Encoding`

The resulting vectors are passed to the first encoder block.

## 5. Multi-Head Self-Attention

Multi-Head Self-Attention helps each token understand its relationship with other tokens in the input sequence.

It helps with the sentence who have ambiguity in them.
Ex: The man saw an astronomer with a telescope   -> this has 2 meanings 

**Example:**

In "The animal didn't cross the road because it was tired", attention can help the model relate *it* to *the animal*.

Multi-head attention uses multiple attention heads to learn different relationships between tokens.

**Output:** Each token receives a context-aware vector.

In the original Transformer, the output dimension remains 512.

## 6. Residual Connection and Layer Normalization

After Multi-Head Attention, the original input is added to the attention output. This is called a residual connection or skip connection.
Formula:
Output = LayerNorm(Input + Attention Output)

<img width="792" height="570" alt="image" src="https://github.com/user-attachments/assets/cafc8131-e75e-410b-945d-078ebf03e2d2" />

Original input ko add kyun karte hain?
Socho attention layer ne useful information learn ki, lekin original input ki information bhi important hai. Addition se original information aage carry ho sakti hai.
Ek aur important benefit: deep network mein gradients ko backpropagation ke dauraan flow karne ka easier path milta hai, jisse training easier ho sakti hai.

After Multi-Head Attention, the original input is added to the attention output. This is called a **residual connection** or skip connection.

**Formula:**

`Output = LayerNorm(Input + Attention Output)`

### Why use residual connections?

- Help gradients flow through deep networks.
- Make deep models easier to train.
- Allow original input information to pass forward.

### Why use Layer Normalization?

- Normalizes feature values within each token representation.
- Helps stabilize training.
- Uses learnable parameters gamma (γ) and beta (β).

## 7. Feed-Forward Neural Network (FFN)

After attention and normalization, each token representation passes through a Feed-Forward Neural Network.

In the original Transformer, it contains two linear layers:

- **First layer:** 512 → 2048 dimensions, followed by ReLU activation.
- **Second layer:** 2048 → 512 dimensions.

**Formula:**

`FFN(x) = ReLU(xW1 + b1)W2 + b2`

The first layer increases the representation's dimension, and the second brings it back to 512.

**Why use FFN?**

It applies additional transformations to each token representation and introduces non-linearity through ReLU.

The same FFN weights are shared across token positions within an encoder block.

## 8. Second Residual Connection and Layer Normalization

After the FFN, the original FFN input is added to its output, followed by Layer Normalization.

**Formula:**

`Final Output = LayerNorm(FFN Input + FFN Output)`

This final output becomes the input to the next encoder block.

## 9. Complete Information Flow

For the original Transformer encoder:

1. Input sentence is tokenized.
2. Tokens are converted into embeddings.
3. Positional encodings are added.
4. Multi-Head Self-Attention captures relationships between tokens.
5. Residual connection and Layer Normalization are applied.
6. Feed-Forward Network transforms each token representation.
7. Another residual connection and Layer Normalization are applied.
8. The output passes to the next encoder block.
9. After all 6 blocks, the final contextual representations are available to the decoder.

**Important:** The original encoder preserves the sequence length and the 512-dimensional representation size throughout its blocks.

## 10. Important Interview Questions

**Q1. What are the two main components of a Transformer encoder block?**

Multi-Head Self-Attention and Feed-Forward Neural Network.

**Q2. Why is positional encoding required?**

To provide information about token positions and sequence order.

**Q3. Why do we use Multi-Head Attention?**

To learn different relationships between tokens using multiple attention heads.

**Q4. What is a residual connection?**

It adds a block's input to its output, helping preserve information and improve gradient flow.

**Q5. Why is Layer Normalization used?**

To normalize features within each token representation and help stabilize training.

**Q6. Why does the FFN expand 512 dimensions to 2048 and then reduce them to 512?**

The larger hidden representation allows additional transformations and non-linear learning while preserving the original representation size at the output.

**Q7. Are all encoder blocks identical?**

Their architecture is the same, but each block has its own learned parameters.

**Q8. What is the output of the encoder?**

Context-aware representations of the input tokens, which are passed to the decoder in the original encoder-decoder Transformer.

## Quick Revision

- **Encoder:** Builds context-aware token representations.
- **Input:** Tokenization + Embedding + Positional Encoding.
- **Attention:** Learns relationships between tokens.
- **Add & Norm:** Residual connection + Layer Normalization.
- **FFN:** 512 → 2048 → 512 in the original Transformer.
- **Original architecture:** 6 encoder blocks.
- **Output:** Context-aware token representations.
