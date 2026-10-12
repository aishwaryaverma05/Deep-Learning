## 3. Self-Attention in Transformers

### 1. What is Self-Attention?

**Self-Attention** is a mechanism in the Transformer architecture that allows each token in an input sequence to focus on other relevant tokens in the same sequence to understand its meaning in context.

It helps the model learn relationships between words, even when they are far apart in a sentence.

**Example:**

"The animal didn't cross the road because it was tired."

To understand the word *"it"*, the model needs to learn its relationship with the word *"animal"*. Self-attention helps the model identify such relationships by considering the context of the entire sentence.

### 2. Why Do We Need Self-Attention?

A word's meaning often depends on the other words in a sentence.

Self-attention helps Transformers to:

- Understand relationships between different tokens.
- Identify which tokens are more relevant to a particular token.
- Capture long-range dependencies between words.
- Generate context-aware representations of tokens.
- Process multiple tokens in parallel during training.

### 3. How Does Self-Attention Work?

Consider the sentence:

**"The cat sat on the mat."**

Self-attention works through the following steps:

#### Step 1: Input Embeddings

First, the input text is divided into tokens. Each token is converted into a numerical vector called an **embedding**.

These embeddings represent the tokens in a form that the neural network can process.

#### Step 2: Create Query, Key, and Value Vectors

For every token, the model creates three vectors:

- **Query (Q):** Represents what information the token is looking for.
- **Key (K):** Helps determine how relevant a token is to another token's Query.
- **Value (V):** Contains the information that can contribute to the final representation.

These vectors are created using learned weight matrices.

#### Step 3: Calculate Attention Scores

The model compares the Query of a token with the Keys of all tokens in the sequence.

These comparisons produce **attention scores**, which indicate how relevant the other tokens are to the token being processed.

For example, when processing "cat", the model calculates its relevance to "The", "sat", "on", "the", and "mat", as well as itself.

#### Step 4: Apply Softmax

The raw attention scores are passed through a function called **Softmax**.

Softmax converts the scores into normalized attention weights that sum to 1.

Higher weights indicate that a token contributes more strongly to the current token's representation.

#### Step 5: Calculate the Weighted Sum of Values

The attention weights are multiplied by their corresponding Value vectors and combined.

This produces a new representation of the token that incorporates relevant information from other tokens.

#### Step 6: Generate Context-Aware Representations

The process is repeated for every token in the sequence.

As a result, each token gets an updated representation that incorporates information from the surrounding context.

These representations are then passed to the next components of the Transformer block.

### 4. Understanding Self-Attention with an Example

Consider the sentence:

**"The cat sat on the mat."**

When the model processes the word **"cat"**:

1. It generates the Query, Key, and Value vectors for all tokens.
2. It compares the Query of "cat" with the Keys of all tokens.
3. It calculates attention weights based on these comparisons.
4. It combines the Value vectors using the calculated weights.
5. It produces a new representation of "cat" that incorporates information from the sentence.

The attention weights are learned from the input and model parameters; they are not manually assigned.

**Important:** The example is illustrative. Actual attention weights depend on the trained model and the input.

### 5. Self-Attention Flow

```text
Input Text
    |
    v
Tokenization
    |
    v
Token Embeddings + Positional Information
    |
    v
Create Query (Q), Key (K), Value (V)
    |
    v
Calculate Attention Scores (Q compared with K)
    |
    v
Apply Softmax to obtain Attention Weights
    |
    v
Weighted Combination of Value (V) Vectors
    |
    v
Context-Aware Token Representations
    |
    v
Further Transformer Processing
```

### 6. Important Points to Remember

- Self-attention is a core mechanism of the Transformer architecture.
- Every token creates its own Query, Key, and Value vectors.
- Attention scores determine the relevance of other tokens to a given token.
- Softmax converts attention scores into normalized weights.
- The weighted sum of Value vectors produces context-aware representations.
- Self-attention can capture relationships between tokens regardless of how far apart they appear in a sequence.
- In standard full self-attention, each token can attend to every token in the input sequence, including itself.
- In decoder-only models such as GPT, **causal self-attention** prevents a token from attending to future tokens during next-token prediction.

### 7. Self-Attention in One Line

**Self-attention allows each token to gather relevant information from other tokens and build a context-aware representation of its meaning.**
