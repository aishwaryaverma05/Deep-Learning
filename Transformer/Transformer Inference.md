# Transformer Inference — Quick Revision Notes

## 1. What is Inference?

**Inference** is the process of using a trained Transformer model to generate predictions on new input data.

Example: English → Hindi translation

Input: `We are friends`

Output: `हम दोस्त हैं`

During inference, the model uses learned weights and biases to generate the output.

## 2. Training vs Inference

| Training | Inference |
|---|---|
| Target sentence is available. | Target sentence is unknown. |
| Decoder receives target tokens shifted right. | Decoder starts with a special `<SOS>` token. |
| Tokens can be processed in parallel using masking. | Tokens are generated one by one. |
| Decoder behaves non-autoregressively in the transcript's explanation. | Decoder behaves autoregressively. |

**Important:** The encoder behaves the same way during training and inference. The main difference is in the decoder.

## 3. Step-by-Step Transformer Inference

### Step 1: Process the input using the Encoder

Input: `We are friends`

1. Tokenize the sentence.
2. Convert tokens into embeddings.
3. Add positional encoding.
4. Apply multi-head self-attention and feed-forward layers, with residual connections and normalization.
5. Obtain contextual vectors for the input tokens.

The encoder output is used by the decoder through cross-attention.

### Step 2: Start the Decoder

The decoder receives a special token called `<SOS>` (Start of Sentence).

- Convert `<SOS>` into an embedding.
- Add positional encoding.
- Pass it through the decoder blocks.

### Step 3: Masked Self-Attention

The decoder applies **Masked Self-Attention** to the tokens generated so far.

- A token cannot attend to future tokens.
- Causal masking is used during inference as well as training.
- This maintains the same autoregressive behavior learned during training.

Attention formula:

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

Here, \(M\) represents the causal mask.

### Step 4: Cross-Attention

The decoder uses the encoder's output to understand the source sentence.

- **Query (Q):** From the decoder.
- **Key (K) and Value (V):** From the encoder.

Cross-attention helps the decoder focus on relevant parts of the input sentence while generating each output token.

### Step 5: Generate the First Token

After the decoder blocks:

1. Apply a Linear layer to produce vocabulary logits.
2. Apply Softmax to obtain probabilities.
3. Select a token according to the decoding strategy.

In the transcript's example, the first predicted token is `हम`.

### Step 6: Feed the Generated Token Back

The next decoder input becomes:

`<SOS> हम`

The decoder processes both tokens, but only the output corresponding to the **last token** is used to predict the next token.

Suppose the next predicted token is `दोस्त`.

The input now becomes:

`<SOS> हम दोस्त`

This process repeats, generating one token at a time.

### Step 7: Stop Generation

The decoder continues generating tokens until it predicts the special `<EOS>` (End of Sentence) token.

Final output:

`हम दोस्त हैं`

## 4. Inference Flow

```text
Input Sentence
      ↓
Encoder
      ↓
Contextual Vectors
      ↓
Decoder receives <SOS>
      ↓
Masked Self-Attention
      ↓
Cross-Attention
      ↓
Feed-Forward Network
      ↓
Linear + Softmax
      ↓
Predict Next Token
      ↓
Append Token to Decoder Input
      ↓
Repeat until <EOS>
```

## 5. Important Points to Remember

- **Autoregressive generation:** Each new token depends on the tokens generated previously.
- **Causal masking:** Prevents the decoder from attending to future positions.
- **Cross-attention:** Connects the decoder to the encoder's representation of the input.
- **Last-token prediction:** Only the final position's output is used to select the next token.
- **EOS token:** Signals that generation should stop.
- **Greedy decoding:** Selects the highest-probability token at each step. Other decoding strategies, such as beam search, are also possible.

## 6. Important Interview Questions

**Q1. What is Transformer inference?**

Inference is the process of using a trained Transformer to generate predictions for new inputs.

**Q2. How does the decoder start generating output?**

It starts with a special `<SOS>` token.

**Q3. What is autoregressive generation?**

Generating one token at a time, using previously generated tokens as input for the next prediction.

**Q4. Is causal masking used during inference?**

Yes. It prevents each position from attending to future positions and preserves autoregressive behavior.

**Q5. What is the role of cross-attention during inference?**

It allows the decoder to use the encoder's output and focus on relevant input tokens.

**Q6. Why do we use Softmax?**

To convert vocabulary logits into a probability distribution over possible next tokens.

**Q7. Why is only the last decoder output used?**

Because we need to predict the next token after the complete sequence generated so far.

**Q8. When does inference stop?**

When the decoder generates `<EOS>` or another configured stopping condition is reached.

**Q9. Does the encoder behave differently during training and inference?**

In the original Transformer translation setup discussed in the transcript, the encoder's processing remains the same.

**Q10. What is the difference between greedy decoding and beam search?**

Greedy decoding selects the highest-probability token at each step. Beam search keeps multiple candidate sequences to explore alternatives.

## Quick Revision

**Input → Encoder → `<SOS>` → Masked Self-Attention → Cross-Attention → FFN → Linear + Softmax → Next Token → Repeat → `<EOS>`**

Remember: **Training processes known target tokens in parallel with masking; inference generates unknown target tokens autoregressively, one at a time.**
