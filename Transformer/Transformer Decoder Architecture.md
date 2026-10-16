# Transformer Decoder Architecture

## 1. What is a Transformer Decoder?

The **Transformer Decoder** generates the output sequence using the encoder's information and previously available output tokens.

**Example: English → Hindi Translation**

- Input: `I am good`
- Output: `मैं अच्छा हूँ`

The encoder processes the English sentence, and the decoder generates the Hindi translation.

## 2. Decoder Architecture Overview

The original Transformer architecture uses **6 decoder blocks stacked together**. Each block contains three main components:

1. Masked Multi-Head Self-Attention
2. Cross-Attention (Encoder–Decoder Attention)
3. Feed-Forward Neural Network (FFN)

Each component is followed by a residual connection and Layer Normalization.

**Architecture flow:**

```text
Target Embeddings + Positional Encoding
                  |
                  v
       Masked Multi-Head Attention
                  |
              Add & Norm
                  |
                  v
             Cross-Attention <--- Encoder Output
                  |
              Add & Norm
                  |
                  v
        Feed-Forward Neural Network
                  |
              Add & Norm
                  |
                  v
         Output to Next Decoder Block
                  |
          (Repeated for 6 blocks)
                  |
                  v
           Linear Layer + Softmax
                  |
                  v
          Next-Token Probabilities
```

*Note: The six blocks have the same architecture, but each block learns its own parameters.*

## 3. Input to the Decoder

The decoder receives the target sentence shifted right during training.

Example:

```text
Actual target:  I am happy <END>
Decoder input:  <START> I am happy
Expected output: I am happy <END>
```

The decoder learns to predict the next token at each position.

Two things are added to create the initial decoder representations:

- **Token Embeddings:** Convert tokens into numerical vectors.
- **Positional Encoding:** Adds information about token positions.

The resulting vectors are passed to the first decoder block.

## 4. Three Main Components

### A. Masked Multi-Head Self-Attention

It allows each target token to attend to earlier target tokens, while preventing it from accessing future tokens.

Example:

```text
Input:  <START> I am
Predict: I, am, happy
```

When predicting `am`, the decoder must not use future target words such as `happy`.

**Purpose:** Prevents information leakage from future tokens during training.

### B. Cross-Attention

Cross-attention connects the decoder with the encoder's output.

| Vector | Source |
|---|---|
| Query (Q) | Decoder representations |
| Key (K) | Encoder output |
| Value (V) | Encoder output |

The decoder uses its Queries to find relevant information in the encoder's Keys and Values.

**Purpose:** Helps the decoder generate output using the original input's context.

### C. Feed-Forward Neural Network (FFN)

After attention, each token representation passes through a small neural network.

In the original Transformer architecture:

```text
Input: 512 dimensions
          |
          v
Dense Layer: 2048 neurons + ReLU
          |
          v
Dense Layer: 512 neurons + Linear
          |
          v
Output: 512 dimensions
```

Formula:

\[
FFN(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2
\]

The same FFN weights are applied independently at every token position within a block.

**Purpose:** Further transforms each token's representation to learn richer patterns.

## 5. What Does Add & Norm Mean?

After each major sublayer, the Transformer uses a residual connection and Layer Normalization.

**Residual connection:**

\[
y=x+\operatorname{Sublayer}(x)
\]

It adds the original input to the sublayer's output, helping preserve information and improve gradient flow.

**Layer Normalization:** Normalizes the feature values of each token representation to help stabilize training.

In the original architecture, the operation is:

```text
Input ───────────────────┐
  |                      |
  v                      |
Attention / FFN          |
  |                      |
  └────── Add ◄──────────┘
           |
           v
       Layer Norm
```

## 6. Output Block: How Is the Next Word Predicted?

After the final decoder block, each token has a contextual representation.

The output block converts these representations into vocabulary scores.

**Step 1: Linear layer**

Projects the final representation to the vocabulary size.

For example, if the vocabulary contains 10,000 tokens, the output produces 10,000 scores per position.

**Step 2: Softmax**

Converts scores into probabilities.

\[
P(w_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

**Step 3: Predict the next token**

The model learns to assign high probability to the correct next token.

Example:

```text
मैं     → 0.10
अच्छा   → 0.75
हूँ     → 0.15
```

The highest-probability token is `अच्छा` in this illustrative example. Actual prediction depends on the model's context and vocabulary.

## 7. Training vs Inference

| Training | Inference |
|---|---|
| Uses the known target sequence, shifted right | Starts with a start token or prompt |
| Uses teacher forcing in standard training | Uses previously generated tokens |
| Positions can be processed in parallel with causal masking | Generates tokens autoregressively |
| Computes loss against the correct target tokens | Stops at an end token or generation limit |

**Important:** The architecture can remain the same while the way tokens are supplied differs. During inference, the decoder generates a token and then uses it as part of the input for the next step.

## 8. Important Interview Questions

**Q1. What is the role of the Transformer Decoder?**

It generates the output sequence using previous target tokens and information from the encoder.

**Q2. What are the three main components of a decoder block?**

Masked Multi-Head Self-Attention, Cross-Attention and Feed-Forward Neural Network.

**Q3. Why is masking required in the decoder?**

It prevents a target position from accessing future target tokens during training.

**Q4. What is the role of Cross-Attention?**

It allows the decoder to focus on relevant information from the encoder output.

**Q5. What is Teacher Forcing?**

During training, the decoder receives the correct previous target tokens rather than its own previous predictions.

**Q6. Why can training be parallelized but generation is usually sequential?**

During training, the target tokens are known and causal masking prevents access to future information, allowing positions to be processed in parallel. During autoregressive inference, the next input token depends on the previous prediction.

**Q7. What is the purpose of the final Linear + Softmax layers?**

The Linear layer produces vocabulary scores, and Softmax converts them into token probabilities.

**Q8. Do all decoder blocks share the same weights?**

No. They share the same architecture, but each block has its own learned parameters.

**Q9. What is the role of FFN in the decoder?**

It applies a nonlinear transformation to each token representation independently, after attention has incorporated contextual information.

**Q10. What is the difference between Masked Self-Attention and Cross-Attention?**

Masked self-attention connects target tokens while blocking future tokens. Cross-attention connects decoder representations with encoder outputs.

## 9. Quick Revision

- The original Transformer has **6 decoder blocks**.
- Each block contains Masked Self-Attention, Cross-Attention and FFN.
- Add & Norm follows each sublayer.
- Decoder input combines token embeddings and positional encoding.
- Masked attention prevents future-token information leakage.
- Cross-attention uses decoder Q and encoder K/V.
- FFN transforms token representations, typically \(512 \rightarrow 2048 \rightarrow 512\) in the original architecture.
- Linear + Softmax produces vocabulary probabilities.
- Training uses known target tokens; autoregressive inference generates tokens step by step.

**Remember:** Masked Self-Attention looks at previous target tokens, Cross-Attention looks at encoder information, and FFN transforms the resulting representations.
