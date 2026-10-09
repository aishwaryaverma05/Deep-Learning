# Artificial Neuron, Weights and Bias

## 1. What Is an Artificial Neuron?

An artificial neuron is the basic computational unit of an Artificial Neural Network (ANN).

It receives input values, multiplies them by their corresponding weights, adds a bias, and passes the result through an activation function to produce an output.

An artificial neuron performs two main operations:

1. Calculate the weighted sum of inputs.
2. Apply an activation function to that sum.

---

## 2. Biological Neuron vs Artificial Neuron

A biological neuron receives signals through dendrites, processes them in the cell body, and transmits signals through its axon.

An artificial neuron is a simplified mathematical model inspired by this process.

| Biological Neuron                  | Artificial Neuron                      |
| ---------------------------------- | -------------------------------------- |
| Dendrites receive signals          | Inputs receive numerical values        |
| Synapses influence signal strength | Weights control input contributions    |
| Cell body processes signals        | Weighted sum and bias are calculated   |
| Neuron sends a signal              | Activation function produces an output |

This is an analogy, not an exact representation of how biological neurons work.

---

## 3. What Are Inputs?

Inputs are the features provided to a neuron.

For example, in a house-price prediction problem, inputs could include:

* House area
* Number of bedrooms
* House age

Suppose the inputs are:

x₁ = 1000 (house area)

x₂ = 3 (number of bedrooms)

x₃ = 5 (house age)

These values are examples only. In real projects, features may need preprocessing and scaling before being passed to a neural network.

---

## 4. What Are Weights?

Weights are learnable parameters that determine how strongly each input contributes to a neuron's weighted sum.

Each input has a corresponding weight.

For example:

* x₁ = 1000, w₁ = 0.5
* x₂ = 3, w₂ = 0.8
* x₃ = 5, w₃ = -0.2

A positive weight contributes positively to the weighted sum, while a negative weight contributes negatively.

A weight's magnitude indicates its influence on the weighted sum, but its practical importance depends on the scales of the features and the other parameters.

During training, the optimizer adjusts weights to help minimize the loss function.

---

## 5. What Is Bias?

Bias is a learnable parameter added to the weighted sum of inputs.

It allows a neuron to shift its activation independently of the input contributions.

Without bias, the weighted sum would be zero whenever all inputs are zero.

For example:

Weighted sum = 10

Bias = 2

Result before activation = 12

Bias is also adjusted during model training.

---

## 6. Mathematical Formula of an Artificial Neuron

The weighted sum is calculated as:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

In compact notation:

z = Σ(wᵢxᵢ) + b

Where:

* xᵢ = input feature
* wᵢ = corresponding weight
* b = bias
* z = weighted sum before activation

The neuron then applies an activation function:

a = f(z)

Here, f represents the activation function and a represents the neuron's output.

For example, with the sigmoid activation function:

sigmoid(z) = 1 / (1 + e⁻ᶻ)

The final output lies between 0 and 1.

---

## 7. Numerical Example

Suppose a neuron receives two inputs:

x₁ = 2

x₂ = 3

Weights:

w₁ = 0.5

w₂ = 0.2

Bias:

b = 1

### Step 1: Multiply each input by its weight

w₁x₁ = 0.5 × 2 = 1

w₂x₂ = 0.2 × 3 = 0.6

### Step 2: Add the weighted inputs

Weighted sum = 1 + 0.6 = 1.6

### Step 3: Add the bias

z = 1.6 + 1 = 2.6

### Step 4: Apply the activation function

Using sigmoid:

a = 1 / (1 + e⁻²·⁶)

a ≈ 0.931

Therefore, the neuron produces an output of approximately 0.931.

This is the output of one neuron, not necessarily the final prediction of an entire neural network.

---

## 8. Python Implementation From Scratch

The following code implements a single artificial neuron using NumPy.

```python
import numpy as np

# Input features
x = np.array([2, 3])

# Weights
w = np.array([0.5, 0.2])

# Bias
b = 1

# Step 1: Calculate weighted sum
z = np.dot(x, w) + b

# Step 2: Apply sigmoid activation
output = 1 / (1 + np.exp(-z))

print("Weighted sum:", z)
print("Neuron output:", output)
```

Expected output:

```text
Weighted sum: 2.6
Neuron output: 0.9308615796566533
```

### Understanding the code

* `np.array()` stores the inputs and weights as arrays.
* `np.dot(x, w)` multiplies corresponding values and adds the products.
* `+ b` adds the bias.
* `np.exp()` calculates the exponential term required by sigmoid.
* `output` stores the neuron's activated output.

This implementation demonstrates a single forward-pass calculation. It does not train the neuron or update its weights.

---

## 9. How Are Weights and Bias Updated During Training?

Initially, weights and biases are generally initialized using suitable initialization methods.

During training:

1. The network makes a prediction.
2. The loss function measures the prediction error.
3. Backpropagation calculates gradients of the loss with respect to weights and biases.
4. The optimizer updates the parameters.

A simplified gradient-descent update is:

w_new = w_old − learning_rate × gradient

b_new = b_old − learning_rate × gradient_of_bias

The process repeats across training iterations so the model can learn patterns from the training data.

---

## 10. Interview Questions and Answers

### Q1. What is an artificial neuron?

An artificial neuron is a mathematical unit that computes a weighted sum of its inputs, adds a bias, and applies an activation function to generate an output.

### Q2. What is the role of weights in an ANN?

Weights determine the contribution of each input to the neuron's weighted sum. They are learned from data during training.

### Q3. Why is bias needed in a neural network?

Bias allows a neuron to shift its activation independently of the weighted inputs, increasing the flexibility of the model.

### Q4. What is the formula for an artificial neuron?

The weighted sum is:

z = Σ(wᵢxᵢ) + b

The output is:

a = f(z)

Here, f is the activation function.

### Q5. What happens if all weights are zero?

For a single neuron, if all weights are zero, its weighted sum becomes equal to its bias. Its output is then the activation function applied to that bias.

If weights are initialized to zero across all neurons in a multilayer network, it can also prevent neurons in the same layer from learning different features because of symmetry. Appropriate weight initialization helps avoid this problem.

### Q6. Can a neuron work without bias?

Yes. A neuron can be constructed without bias, but it has less flexibility because its weighted sum cannot be shifted independently of its inputs.

### Q7. What is the difference between a weight and a bias?

A weight scales an input's contribution, whereas bias is an additional learnable term that shifts the weighted sum.

### Q8. Are weights and biases fixed?

No. They are generally learnable parameters updated during training using gradients and an optimizer.

### Q9. Does a weight's positive or negative sign matter?

Yes. A positive weight contributes positively to the weighted sum, while a negative weight contributes negatively. However, the final effect on the output also depends on the activation function and other parameters.

### Q10. What is the difference between a weighted sum and a neuron's output?

The weighted sum is calculated before applying the activation function. The neuron's output is obtained after applying the activation function to that weighted sum.

### Q11. What happens if the bias is very large?

A large positive or negative bias can shift the neuron's activation substantially. Depending on the activation function, this may cause saturation or make learning less effective.

### Q12. How are weights and biases learned?

Backpropagation calculates their gradients, and an optimizer uses those gradients to update the parameters to reduce the loss.

---

## 11. Quick Revision

Remember this sequence:

**Inputs → Multiply by weights → Add bias → Apply activation function → Output**

Key formulas:

Weighted sum:

z = Σ(wᵢxᵢ) + b

Neuron output:

a = f(z)

A neuron performs the forward calculation. Training adjusts its learnable parameters so the overall network can improve its predictions.

---

## 12. Practical Exercise

Modify the Python code and experiment with the following:

1. Change the bias from 1 to 0.
2. Change one weight from 0.2 to -0.2.
3. Replace sigmoid with ReLU: `max(0, z)`.
4. Observe how the output changes.
5. Explain why the output changes in each case.

These experiments will help you understand the role of each parameter rather than memorizing the formulas.
