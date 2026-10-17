<img width="852" height="732" alt="image" src="https://github.com/user-attachments/assets/9b30adc6-2b6d-41a4-833e-81342204a3eb" />
# LLM Fundamentals — Quick Revision Notes

## 1. What is an LLM?

**LLM (Large Language Model)** is a deep learning model trained on a large amount of text data to understand and generate human-like text.

LLMs learn patterns, relationships between words, and language structure during training.

**Examples:** GPT, Llama, Gemini, Claude.

**Applications:**
- Text generation and summarization
- Question answering
- Code generation
- Translation and sentiment analysis
- Chatbots and AI assistants

## 2. What is a Token?

A **token** is a small unit of text processed by an LLM. It can be a word, part of a word, punctuation, or another text unit.

Example:

`"I love machine learning!"`

A tokenizer might split it into:

`["I", " love", " machine", " learning", "!"]`

The exact tokens depend on the tokenizer.

**Important points:**
- LLMs process tokens, not directly human-readable sentences.
- One word can contain multiple tokens.
- Tokenization converts text into token IDs that the model can process.
- Token count affects API cost and context-window usage.

## 3. What are Embeddings?

**Embeddings** are numerical vectors that represent tokens or other data in a way that captures useful patterns and relationships.

Example (illustrative):

- `king → [0.21, 0.75, -0.12]`
- `queen → [0.19, 0.71, -0.08]`
- `car → [-0.45, 0.13, 0.82]`

These numbers are simplified examples, not actual model embeddings.

**Why are embeddings useful?**
- Represent text numerically for neural networks.
- Help Transformers process token representations.
- Enable semantic similarity search in RAG systems.
- Help represent meaning beyond exact keyword matches.

**Remember:** Token IDs are identifiers; embeddings are learned numerical representations. Similar meanings may have similar embeddings, depending on the model and training.

## 4. What is a Context Window?

The **context window** is the maximum amount of tokenized input and other supported context that a model can consider in a single request.

Depending on the model and API, it can include:
- System instructions
- Conversation history
- User's current prompt
- Retrieved documents
- Tokens generated in the response

**Example:** If a model supports a 32,000-token context window, the combined request context and generated output must fit within the applicable limit.

**Why does it matter?**
- Longer documents and conversations consume more tokens.
- Older information may need to be omitted or summarized when limits are reached.
- Larger context windows do not guarantee perfect recall of every detail.

## 5. How Does an LLM Generate Text?

Most modern generative LLMs use **next-token prediction**.

The basic process:

1. Receive the input prompt.
2. Tokenize the input into token IDs.
3. Process the tokens through the model.
4. Calculate scores (logits) for possible next tokens.
5. Convert scores into probabilities.
6. Select the next token using the decoding strategy.
7. Append the token and repeat until a stopping condition is reached.

**Example:**

Prompt: `The capital of France is`

The model may assign a high probability to `Paris` and generate it as the next token.

The model continues predicting tokens until it produces an answer or reaches a stopping condition.

**Important:** Next-token prediction does not mean the model simply copies text. It learns statistical patterns from training data and uses them to generate new sequences.

## 6. What is Model Inference?

**Inference** is the process of using a trained model to produce predictions or outputs for new inputs.

For an LLM, inference generally involves two stages:

### A. Prefill
The model processes the input prompt and builds the internal representations needed for generation.

### B. Decode
The model generates output tokens one by one, using the preceding context and previously generated tokens.

**Example:**

Input: `Explain artificial intelligence in simple words.`

The model processes the prompt during prefill, then generates the answer during decoding.

**Training vs. Inference**

| Training | Inference |
|---|---|
| Learns model parameters from data | Uses learned parameters to generate outputs |
| Updates weights during optimization | Usually keeps weights fixed |
| Computationally expensive overall | Cost depends on model size, prompt and output length |

## 7. What Are Parameters and Weights?

**Parameters** are the learned numerical values inside a model, including weights and biases.

During training, optimization adjusts these parameters to reduce the training objective.

A model with more parameters may have greater capacity, but parameter count alone does not determine quality. Architecture, training data, training methods and inference setup also matter.

## 8. What is Temperature?

**Temperature** controls the randomness of token selection by adjusting the model's output probability distribution.

- **Low temperature:** More predictable, often more consistent outputs.
- **High temperature:** More varied outputs, potentially less predictable.

Temperature does not retrain the model or directly change its learned knowledge.

The exact effect depends on the model and decoding configuration.


Low Temperature (e.g., 0.0 to 0.3)

• What it means: The AI plays it very safe and acts like a strict robot. It almost always picks the word with the highest probability.
• Best used for: Math, coding, data extraction, or factual Q&A where you need the exact same, correct answer every time.
• Example:
	• Prompt: "The capital of France is..."
	• Low Temperature Output: "Paris." (It picks the most obvious, logical word every single time).

Medium Temperature (e.g., 0.7)

• What it means: A balanced mix. It usually picks sensible words, but occasionally mixes in a slightly different choice to keep things natural.
• Best used for: General chat, emails, and normal summaries.
• Example:
	• Prompt: "Write a polite greeting for an email."
	• Medium Temperature Output: "Dear John, I hope you are having a wonderful week." (Natural and smooth, with a little variety).

High Temperature (e.g., 1.0 to 1.5+)

• What it means: The AI gets adventurous and takes risks. It flattens the odds so that less-likely, wild, or unusual words have a much better chance of getting picked.
• Best used for: Creative writing, brainstorming story ideas, or writing poetry.
• Example:
	• Prompt: "Describe the sky."
	• High Temperature Output: "A bruised canvas of violet smoke and liquid amber bleeding into the ocean." (Very creative, colorful, and unpredictable).

## 9. Limitations of LLMs

- **Hallucination:** Can generate plausible-sounding but incorrect information.
- **Knowledge limitations:** May lack recent or private information.
- **Context limitations:** Cannot necessarily retain every detail in long inputs.
- **Bias:** May reproduce biases present in training data.
- **Nondeterminism:** Outputs may vary with sampling settings or system configuration.

**How can we improve reliability?**
- Write clear prompts.
- Provide trusted context using RAG.
- Validate outputs and structured data.
- Use tools or external sources when appropriate.
- Test important answers rather than blindly trusting them.

## 10. Important Interview Questions

**Q1. What is an LLM?**

An LLM is a deep learning model trained on large text datasets to learn language patterns and perform tasks such as text generation, summarization and question answering.

**Q2. What is the difference between a token and an embedding?**

A token is a unit of text represented by a token ID. An embedding is a numerical vector representing that token or another piece of data.

**Q3. What is next-token prediction?**

It is the task of estimating the probability distribution of the next token given the preceding tokens.

**Q4. What is a context window?**

It is the maximum amount of tokenized context a model can handle in a request, subject to its specific limits.

**Q5. What is the difference between training and inference?**

Training learns model parameters from data; inference uses the trained parameters to generate predictions or responses.

**Q6. Why do LLMs hallucinate?**

They generate outputs based on learned patterns and available context, which can produce plausible answers without factual verification.

**Q7. Does a larger model always perform better?**

No. Performance also depends on architecture, training data, training quality, fine-tuning, inference and the task.

**Q8. What is the role of temperature?**

It adjusts the probability distribution used for token selection, influencing output variability.

**Q9. Why do LLMs need tokenization?**

Tokenization converts text into units that can be mapped to numerical IDs and processed by the model.

**Q10. What is the difference between an LLM and an embedding model?**

An LLM is generally designed to process and generate language. An embedding model converts inputs into vector representations, often for semantic search or similarity tasks.

---

## Quick Revision

- **LLM:** Learns language patterns and generates text.
- **Token:** Unit of text processed by the model.
- **Embedding:** Numerical representation of a token or data item.
- **Context window:** Limit on the context processed in a request.
- **Next-token prediction:** Core generation objective of many LLMs.
- **Inference:** Using a trained model to generate outputs.
- **Parameters:** Learned numerical values of the model.
- **Temperature:** Controls variability in token selection.
- **Hallucination:** Plausible output that may be factually wrong.

**Next topic:** Generative AI — Prompt Engineering, Model APIs and Structured Outputs.
