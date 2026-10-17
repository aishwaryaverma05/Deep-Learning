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

{Attention}(Q,K,V)={softmax}((QK^T)*V/(d_k))


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
