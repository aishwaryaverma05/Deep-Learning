# Masked Self-Attention — Transformer Decoder

## 1. What is Masked Self-Attention?

Masked Self-Attention is a type of self-attention used in the Transformer decoder that prevents a token from attending to future tokens.

**Main purpose:** The model should use only the current and previous tokens, not future tokens, while predicting the next token.

## 2. Why Do We Need Masking?

Consider the sentence:

`You are happy`

When predicting the next token:

- For `You`, the model should not see `are` or `happy`.
- For `are`, the model can see `You` and `are`, but not `happy`.
- For `happy`, the model can see all three tokens.

Without masking, a token could use future-token information during training, causing **data leakage**.

## 3. What is Autoregressive Generation?

An autoregressive model generates one token at a time, using previously generated tokens to predict the next one.

Example:

`I → I am → I am happy`

During inference, the decoder generates tokens sequentially because future tokens are not yet available.

## 4. Why Is Training Different from Inference?

### During inference
- The model generates one token at a time.
- The next prediction depends on previously generated tokens.
- Generation is autoregressive.

### During training
- The correct target sequence is already available.
- With teacher forcing, the correct previous tokens are provided as decoder inputs.
- Tokens can be processed in parallel, making training faster.
- However, future-token information must be blocked to prevent data leakage.

## 5. How Does the Mask Work?

Self-attention calculates attention scores using Query (Q) and Key (K):

`Attention Scores = QKᵀ / √dₖ`

Before applying Softmax, a **causal mask** is added to the attention scores.

- Allowed positions: mask value is `0`.
- Future positions: mask value is `−∞`.

Example mask for three tokens:

```text
        You   are   happy
You      0    -∞     -∞
are      0     0     -∞
happy    0     0      0
```

After adding the mask, Softmax converts the scores at future positions to zero probability.

Therefore:
- `You` attends only to `You`.
- `are` attends to `You` and `are`.
- `happy` can attend to all three tokens.

**Important:** Masking does not remove future tokens from the input. It prevents attention from using their information.

## 6. Why Is Masking Applied Before Softmax?

The masked attention formula is:

`Masked Attention = Softmax((QKᵀ / √dₖ) + Mask)V`

Adding `−∞` to a future-token score makes it `−∞`. Softmax then assigns that position a probability of zero.

As a result, the future token does not contribute to the current token's contextual representation.

## 7. How Does Masking Solve Both Problems?

Without masking:
- Parallel training is possible.
- But future information can leak into earlier token representations.

With sequential training:
- Future information is not leaked.
- But training becomes slower because tokens are processed step by step.

**Masked Self-Attention provides both benefits:**
- Parallel processing during training.
- No access to future tokens.
- Autoregressive generation during inference.

## 8. Quick Revision

| Concept | Meaning |
|---|---|
| Self-Attention | Learns relationships between tokens |
| Masked Self-Attention | Prevents attention to future tokens |
| Autoregressive | Generates the next token using previous tokens |
| Teacher Forcing | Uses correct previous target tokens during training |
| Causal Mask | Blocks future positions in attention |
| `−∞` before Softmax | Makes masked positions have zero attention probability |

## 9. Important Interview Questions

**Q1. What is Masked Self-Attention?**

It is self-attention that prevents a token from attending to future tokens.

**Q2. Why is masking required in a Transformer decoder?**

To prevent future-token information from leaking into the prediction of the current token.

**Q3. How does the causal mask work?**

It adds `−∞` to future-position attention scores before Softmax, giving those positions zero probability.

**Q4. Why can Transformer decoder training be parallelized?**

The complete target sequence is available during training, so the decoder can process token positions together while masking future information.

**Q5. Why is inference autoregressive?**

The next token must be generated using the tokens already available; future tokens are not known yet.

**Q6. What is the difference between Self-Attention and Masked Self-Attention?**

Regular self-attention can attend to all positions in its input, whereas masked self-attention blocks future positions.

---

### One-Line Summary

**Masked Self-Attention allows parallel training while preventing each token from accessing future-token information.**
