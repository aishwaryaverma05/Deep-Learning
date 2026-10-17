# Normalization in Deep Learning

## 1. What is Normalization?

Normalization is a technique used to bring input data or values into a similar scale or range.

In Deep Learning, it helps make the training process smoother and more stable.

**Example:** If one feature has values between 0–1 and another has values between 0–10000, normalization can bring them to a comparable scale.

## 2. Why Do We Use Normalization?

We use normalization to:
- Make training faster and more stable.
- Help the model learn more efficiently.
- Prevent features with large values from dominating smaller ones.
- Make optimization easier.

## 3. Benefits of Normalization

- **Faster training:** The model may learn faster.
- **Stable learning:** Training becomes more consistent.
- **Better optimization:** The optimizer can update weights more effectively.
- **Improved performance:** It may help the model perform better, although improvement is not guaranteed.

## 4. What is Internal Covariate Shift?

Internal Covariate Shift is the term originally used to describe changes in the distribution of a layer's inputs during training as the parameters of previous layers change.

In simple words, when earlier layers learn and update their weights, the input received by the next layer can keep changing.

## 5. What Problems Can It Cause?

These changing inputs can make training harder because:
- Later layers have to adapt to changing inputs.
- Training may become slower or less stable.
- Choosing suitable learning rates can become more difficult.

**Note:** Internal covariate shift was the original motivation for Batch Normalization. However, later research suggests that reducing internal covariate shift is not necessarily the main reason Batch Normalization works.

## 6. What is Batch Normalization?

Batch Normalization (BatchNorm) is a technique that normalizes the activations of a layer using statistics calculated from a mini-batch of training examples.

It then uses learnable parameters called **gamma (γ)** and **beta (β)** to scale and shift the normalized values.

### How It Works

1. Calculate the mean of the mini-batch.
2. Calculate the variance of the mini-batch.
3. Normalize the values using the mean and variance.
4. Apply learnable scale and shift parameters.

**Formula:**

`x_norm = (x - mean) / sqrt(variance + epsilon)`

`output = gamma * x_norm + beta`

**Key Point:** Batch Normalization uses mini-batch statistics during training. During inference, it typically uses running estimates of the mean and variance.

## 7. How Does Layer Normalization Work?

Layer Normalization (LayerNorm) normalizes values across the features of each individual example.

Unlike Batch Normalization, it does not calculate normalization statistics across the batch.

### Steps

1. Take the feature values of one example.
2. Calculate their mean and variance.
3. Normalize the values.
4. Apply learnable scale (gamma) and shift (beta) parameters.

## 8. Explain Layer Normalization

Suppose one example has three feature values:

`[2, 4, 6]`

- Mean = 4
- Variance = 2.67 (approximately)
- Each value is normalized using the mean and variance.
- Learnable scale and shift parameters are then applied.

The result is a normalized set of values with approximately zero mean and unit variance before the scale and shift.

### Batch Normalization vs Layer Normalization

| Batch Normalization | Layer Normalization |
|---|---|
| Normalizes using statistics across a mini-batch. | Normalizes across features of each example. |
| Depends on batch statistics during training. | Works independently for each example. |
| Commonly used in CNNs. | Commonly used in Transformers and NLP models. |
| Uses running statistics during inference. | Does not require running batch statistics. |

**Quick Revision:**
- **Normalization:** Brings values to a suitable scale or distribution.
- **Internal Covariate Shift:** The original term for changes in layer-input distributions during training.
- **BatchNorm:** Normalizes using mini-batch statistics.
- **LayerNorm:** Normalizes across features within each example.
- **Gamma and Beta:** Learnable parameters used to scale and shift normalized values.
