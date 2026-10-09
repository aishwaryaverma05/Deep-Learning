# Activation Functions in Artificial Neural Networks

## 1. What Is an Activation Function?

An activation function is a mathematical function applied to a neuron's weighted sum to calculate its output.

First, a neuron calculates:

z = Σ(wᵢxᵢ) + b

Then the activation function is applied:

a = f(z)

Where:

* z = weighted sum plus bias
* f = activation function
* a = output of the neuron

## 2. Why Do We Need Activation Functions?

Activation functions introduce non-linearity into a neural network.

Without non-linear activation functions, stacking multiple ordinary linear layers is mathematically equivalent to a single linear transformation. Such a network cannot learn arbitrary non-linear relationships just by adding more linear layers.

Activation functions help neural networks learn complex patterns in data.

## 3. Types of Activation Functions

### A. Linear Activation Function

Formula:

f(z) = z

The output is equal to the input.

Characteristics:

* Output range: negative infinity to positive infinity.
* Commonly used in the output layer of regression models.
* Does not introduce non-linearity.

Example:

If z = 5, then f(z) = 5.

### B. Sigmoid Activation Function

Formula:

sigmoid(z) = 1 / (1 + e⁻ᶻ)

Output range: 0 to 1.

Characteristics:

* Converts the input into a value between 0 and 1.
* Commonly used in the output layer of binary classification models.
* Can suffer from vanishing gradients for very large positive or negative inputs.

Example:

If z = 0, sigmoid(z) = 0.5.

A sigmoid output can be interpreted as a predicted probability when the model is appropriately trained and used for binary classification.

### C. Tanh Activation Function

Formula:

tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)

Output range: -1 to 1.

Characteristics:

* Produces negative, zero, and positive outputs.
* Is zero-centred.
* Can suffer from vanishing gradients when the input magnitude is large.

Example:

If z = 0, tanh(z) = 0.

### D. ReLU (Rectified Linear Unit)

Formula:

ReLU(z) = max(0, z)

Output range: 0 to positive infinity.

Characteristics:

* Returns zero for negative inputs.
* Returns the input itself for positive inputs.
* Computationally simple.
* Commonly used in hidden layers.
* Can suffer from the dying ReLU problem when units consistently receive negative inputs.

Examples:

If z = -3, ReLU(z) = 0.

If z = 4, ReLU(z) = 4.

### E. Leaky ReLU

Formula:

f(z) = z, when z > 0

f(z) = αz, when z ≤ 0

Here, α is a small positive constant, such as 0.01.

Characteristics:

* Allows a small non-zero output for negative inputs.
* May help reduce the dying ReLU problem.
* Its performance depends on the task and implementation.

Example:

If z = -3 and α = 0.01, then f(z) = -0.03.

### F. Softmax Activation Function

For class i:

softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ

Characteristics:

* Converts a vector of logits into values between 0 and 1 that sum to 1.
* Commonly used in the output layer of multiclass classification models.
* Each output represents a class score interpreted as a probability under the model.

Example:

For a three-class problem, a softmax output might be:

[0.70, 0.20, 0.10]

The values sum to 1, and the first class has the highest predicted probability.

## 4. Comparison of Activation Functions

| Function   | Output Range                        | Common Usage                                   |
| ---------- | ----------------------------------- | ---------------------------------------------- |
| Linear     | (-∞, +∞)                            | Regression output                              |
| Sigmoid    | (0, 1)                              | Binary classification output                   |
| Tanh       | (-1, 1)                             | Some hidden layers and recurrent architectures |
| ReLU       | [0, +∞)                             | Hidden layers                                  |
| Leaky ReLU | (-∞, +∞)                            | Hidden layers                                  |
| Softmax    | Each output in (0, 1), summing to 1 | Multiclass classification output               |

These are common patterns, not strict rules for every architecture.

## 5. Python Implementation Using NumPy

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

z = np.array([-3, -1, 0, 1, 3])

print("Input:", z)
print("Sigmoid:", sigmoid(z))
print("ReLU:", relu(z))
print("Tanh:", tanh(z))
print("Leaky ReLU:", leaky_relu(z))
```

The same input produces different outputs depending on the activation function.

## 6. Using Activation Functions in TensorFlow/Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential([
    Input(shape=(30,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.summary()
```

In this example:

* `Input(shape=(30,))` accepts 30 input features.
* The hidden layers use ReLU.
* The output layer uses sigmoid for binary classification.

This builds the model architecture; it does not train the model.

## 7. How to Choose an Activation Function

### Binary Classification

A common setup is:

* Hidden layers: ReLU
* Output layer: Sigmoid
* Loss: Binary cross-entropy

### Multiclass Classification

A common setup is:

* Hidden layers: ReLU
* Output layer: Softmax
* Loss: Categorical cross-entropy or sparse categorical cross-entropy, depending on label format.

### Regression

A common setup is:

* Hidden layers: ReLU
* Output layer: Linear activation, or no explicit activation.

The final choice depends on the target values, model architecture, and loss function.

## 8. Interview Questions and Answers

### Q1. What is an activation function?

An activation function transforms a neuron's weighted sum into an output. Non-linear activation functions enable neural networks to learn non-linear relationships.

### Q2. Why can't we use only linear activation functions in a deep neural network?

A composition of linear transformations is itself a linear transformation. Without non-linearity, adding more ordinary linear layers does not increase the network's ability to represent non-linear relationships.

### Q3. What is the difference between sigmoid and tanh?

Sigmoid produces outputs between 0 and 1, whereas tanh produces outputs between -1 and 1. Tanh is zero-centred, while sigmoid is not.

### Q4. Why is ReLU widely used in hidden layers?

ReLU is computationally simple and provides a non-linear transformation with a non-zero gradient for positive inputs. However, it can produce inactive units for negative inputs.

### Q5. What is the dying ReLU problem?

A ReLU neuron can become inactive if it consistently receives negative inputs, producing zero outputs and zero gradients through that activation. Leaky ReLU is one possible alternative.

### Q6. What is the vanishing gradient problem?

During backpropagation, gradients can become very small as they propagate through layers or time steps. This can make early layers learn very slowly. Saturating activation functions such as sigmoid and tanh can contribute to this problem.

### Q7. Why is sigmoid often used for binary classification?

Sigmoid converts a single logit into a value between 0 and 1, which can be used as a predicted probability for the positive class.

### Q8. Why is softmax used for multiclass classification?

Softmax converts multiple class logits into a normalized distribution of class probabilities, making it useful when classes are mutually exclusive.

### Q9. Can ReLU be used in an output layer?

Yes, when the output needs to be non-negative and unbounded, such as certain regression tasks. However, it is not the usual choice for binary probability outputs.

### Q10. Which activation function is best?

There is no universally best activation function. ReLU is a common starting point for hidden layers, while output activations are chosen according to the task.

## 9. Quick Revision

* Activation functions transform a neuron's weighted sum.
* Non-linear activation functions allow neural networks to learn complex relationships.
* ReLU is a common choice for hidden layers.
* Sigmoid is common for binary classification outputs.
* Softmax is common for mutually exclusive multiclass classification.
* Linear activation is common for regression outputs.
* Saturating functions can contribute to vanishing gradients.

## 10. Practical Exercise

1. Run the NumPy code.
2. Change the input values.
3. Observe how sigmoid, tanh, ReLU, and Leaky ReLU respond.
4. Explain why ReLU returns zero for negative inputs.
5. Build a small Keras model and identify which activation function is used in each layer.
