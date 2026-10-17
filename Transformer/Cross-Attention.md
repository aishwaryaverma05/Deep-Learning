# Cross-Attention in Transformers

## 1. What is Cross-Attention?

**Cross-Attention** is a mechanism that helps a model find relationships between two different sequences.

It allows the decoder to focus on relevant parts of the input sequence while generating the output sequence.

**Example: English → Hindi Translation**

- Input: `I like eating ice cream`
- Output: `मुझे आइसक्रीम खाना पसंद है`

While generating a Hindi word, the decoder can focus on the relevant English words using cross-attention.

## 2. Why Do We Need Cross-Attention?

The decoder needs two types of information to generate the next word:

1. **Previous output words:** What has the decoder generated so far?
2. **Input sequence:** What information was present in the original input?

Masked Self-Attention helps the decoder understand previous output words. Cross-Attention helps it connect the output with the original input.

For example:
- `पसंद` ↔ `like`
- `आइसक्रीम` ↔ `ice cream`
- `खाना` ↔ `eating`

Cross-attention learns these relationships through attention weights.

## 3. How Does Cross-Attention Work?

Cross-attention follows a process similar to self-attention, but its Query, Key and Value vectors come from different sequences.

### Step 1: Generate embeddings

We have two sequences:

- Input sequence: `We are friends`
- Output sequence: `हम दोस्त हैं`

Each token has an embedding or contextual embedding.

### Step 2: Generate Q, K and V

This is the **most important difference** from self-attention.

| Vector | Comes from |
|---|---|
| Query (Q) | Decoder / output sequence |
| Key (K) | Encoder / input sequence |
| Value (V) | Encoder / input sequence |

The decoder's Query asks: *Which information from the input is relevant to the current output token?*

The Keys help calculate relevance, and the Values carry the information used to build the output representation.

### Step 3: Calculate attention scores

Queries are compared with Keys using the dot product.

\[
\text{Scores}=\frac{QK^T}{\sqrt{d_k}}
\]

Here, \(d_k\) is the dimension of the Key vectors.

### Step 4: Apply Softmax

\[
\text{Attention Weights}=
\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
\]

Softmax converts scores into attention weights, representing how much attention each output token gives to each input token.

### Step 5: Calculate the weighted sum

\[
\text{Output}=\text{Attention Weights}\times V
\]

The weighted sum of the input Value vectors produces contextual representations for the output tokens.

## 4. Cross-Attention vs Self-Attention

| Feature | Self-Attention | Cross-Attention |
|---|---|---|
| Input | One sequence | Two sequences |
| Query | Same sequence | Decoder/output sequence |
| Key | Same sequence | Encoder/input sequence |
| Value | Same sequence | Encoder/input sequence |
| Main purpose | Understand relationships within a sequence | Connect input and output sequences |
| Output vectors | One per input token | One per output token |

**Quick memory trick:**

- Self-Attention: Q, K, V come from the **same sequence**.
- Cross-Attention: Q comes from the **decoder**, while K and V come from the **encoder**.

## 5. Where Is Cross-Attention Used?

- **Machine translation:** English → Hindi.
- **Question answering:** Connect a question with relevant information in a context.
- **Image captioning:** Connect image features with generated text.
- **Text-to-image generation:** Connect text representations with image-generation representations.
- **Text-to-speech:** Connect text with generated speech representations.

Cross-attention is especially useful when a model needs to connect information from different sequences or modalities.

## 6. Important Interview Questions

**Q1. What is Cross-Attention?**

Cross-attention allows a model to relate two different sequences. In a Transformer decoder, it helps the decoder focus on relevant information from the encoder output.

**Q2. What is the main difference between Self-Attention and Cross-Attention?**

Self-attention calculates relationships within the same sequence, whereas cross-attention calculates relationships between two sequences.

**Q3. Where do Q, K and V come from in Cross-Attention?**

Q comes from the decoder's representations, while K and V come from the encoder's output representations.

**Q4. Why is Cross-Attention needed in a Transformer decoder?**

It allows the decoder to use information from the input sequence while generating the output sequence.

**Q5. What is the role of the attention weights?**

They determine how strongly each output token attends to the different input tokens.

**Q6. How many output vectors does Cross-Attention produce?**

It produces one output vector per output-sequence token, assuming the standard cross-attention setup.

**Q7. What is the Cross-Attention formula?**

\[
\text{Attention}(Q,K,V)=
\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**Q8. What is the difference between Masked Self-Attention and Cross-Attention in the decoder?**

Masked self-attention lets the decoder use previous output tokens without seeing future tokens. Cross-attention lets the decoder access the encoder's input representations.

## 7. Quick Revision

- Cross-attention connects **two different sequences**.
- In the Transformer decoder, **Q comes from the decoder; K and V come from the encoder**.
- Attention scores measure the relevance of input tokens to output tokens.
- Softmax converts scores into attention weights.
- The weighted sum of V produces contextual output representations.
- It is used in translation, question answering and multimodal tasks.

**Remember:** Self-attention understands relationships within a sequence; cross-attention connects the decoder's output sequence to the encoder's input sequence.
