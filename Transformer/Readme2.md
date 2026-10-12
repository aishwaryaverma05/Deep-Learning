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



# Scaled Dot-Product Attention

## 1. What is Scaled Dot-Product Attention?

Scaled Dot-Product Attention is the mechanism used in Transformers to determine **how much attention one token should pay to other tokens** in a sequence.

It uses three vectors:
- **Query (Q):** What information is this token looking for?
- **Key (K):** What information does each token offer?
- **Value (V):** What actual information should be collected from each token?

It calculates the similarity between Queries and Keys, converts the scores into attention weights, and uses these weights to combine the Values.

**In simple words:** It helps a token identify which other tokens are important for understanding its context.

## 2. Why is it called Scaled Dot-Product Attention?

The name comes from two main operations:

- **Dot Product:** Measures how closely a Query matches a Key.
- **Scaling:** Divides the scores by the square root of the Key vector's dimension to prevent them from becoming too large.

After these operations, Softmax converts the scores into attention weights.

## 3. Architecture / Working

The process follows these steps:

```text
Query (Q) ──┐
            ├──> Dot Product (Q × Kᵀ)
Key (K) ────┘
                      |
                      v
              Scale the scores
               by √dₖ
                      |
                      v
                  Softmax
                      |
                      v
              Attention Weights
                      |
                      v
Value (V) ─────> Weighted Sum
                      |
                      v
                Final Output
```

### Step 1: Calculate the Dot Product

The Query of a token is compared with the Keys of all relevant tokens.

A higher score generally indicates a stronger match between the Query and a Key.

### Step 2: Scale the Scores

The dot-product scores are divided by \(\sqrt{d_k}\), where \(d_k\) is the dimension of the Key vectors.

**Why is scaling needed?**

When vector dimensions are large, dot-product scores can become large. This can make Softmax produce extremely concentrated probabilities, resulting in very small gradients during training.

Scaling helps keep the scores at a more suitable magnitude.

### Step 3: Apply Softmax

Softmax converts the scaled scores into attention weights.

- Each weight lies between 0 and 1.
- The weights across the relevant tokens sum to 1.
- A larger weight means that the corresponding Value contributes more to the output.

### Step 4: Multiply by Values

Each Value vector is multiplied by its attention weight. The weighted Values are then added together to produce the final attention output.

This output contains information gathered from the relevant tokens.

## 4. Mathematical Formula

The complete formula is:

\[
\boxed{\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}
\]

Where:

- \(Q\) = Query matrix
- \(K\) = Key matrix
- \(K^T\) = Transpose of the Key matrix
- \(V\) = Value matrix
- \(d_k\) = Dimension of each Key vector
- Softmax = Converts scores into attention weights

## 5. Simple Example

Consider the sentence:

**"The animal didn't cross the road because it was tired."**

To understand the word *"it"*, the model needs to identify which other word provides relevant context.

Suppose the Query for *"it"* produces the following illustrative attention weights:

| Token | Attention Weight |
|---|---:|
| animal | 0.70 |
| road | 0.10 |
| tired | 0.15 |
| Other tokens | 0.05 |

These values are only an illustration, not actual model outputs.

The model assigns more weight to *"animal"* in this example. Its Value therefore contributes more strongly to the resulting representation of *"it"*.

**Important:** Attention weights indicate how information is combined; they do not always provide a definitive explanation of a model's reasoning.

## 6. Why Do Transformers Use It?

- Helps capture relationships between tokens.
- Allows a token to use information from other relevant tokens.
- Supports contextual representations, where a word's representation depends on its surrounding text.
- Provides the foundation for Multi-Head Attention in Transformers.

## 7. Key Takeaways

- **Query:** What am I looking for?
- **Key:** How relevant am I to that query?
- **Value:** What information do I contribute?
- **Dot Product:** Calculates Query-Key compatibility scores.
- **Scaling:** Divides scores by \(\sqrt{d_k}\).
- **Softmax:** Converts scores into attention weights.
- **Weighted Sum:** Combines Values to generate the output.

**In one line:** Scaled Dot-Product Attention compares Queries with Keys, scales and normalizes the scores, and uses the resulting weights to combine Values.
