# Forward Propagation in Artificial Neural Networks

## 1. What Is Forward Propagation?

Forward propagation is the process of passing input data through a neural network, layer by layer, to calculate the final prediction.

During forward propagation:

1. Input features enter the network.
2. Each neuron calculates a weighted sum and adds a bias.
3. An activation function transforms the result.
4. The output of one layer becomes the input to the next layer.
5. The final layer generates the prediction.

Forward propagation calculates predictions. It does not update the model's weights or biases.

---

## 2. How Does Forward Propagation Work?

Consider a neural network with:

* Two input features
* One hidden layer with two neurons
* One output neuron

The information flows as follows:

Input → Hidden Layer → Output Layer → Prediction

### Step 1: Input Layer

Suppose the input features are:

x₁ = 2

x₂ = 3

These values are passed to the hidden layer.

### Step 2: Calculate the Hidden Layer

Suppose the first hidden neuron has:

w₁₁ = 0.5

w₂₁ = 0.2

b₁ = 0.1

Its weighted sum is:

z₁ = (x₁ × w₁₁) + (x₂ × w₂₁) + b₁

z₁ = (2 × 0.5) + (3 × 0.2) + 0.1

z₁ = 1.7

Using ReLU:

a₁ = max(0, 1.7) = 1.7

The first hidden neuron produces an output of 1.7.

The second hidden neuron performs a similar calculation using its own weights and bias.

### Step 3: Calculate the Output Layer

The hidden-layer outputs are passed to the output neuron.

The output neuron calculates another weighted sum, adds its bias, and applies its activation function.

For binary classification, the output layer commonly uses sigmoid to produce a value between 0 and 1.

### Step 4: Generate the Prediction

The final sigmoid output can be interpreted as the predicted probability of the positive class when the model is appropriately trained.

For example, if the output is 0.85, the model predicts an 85% probability for the positive class.

The classification threshold determines how this probability is converted into a class label.

---

## 3. Mathematical Representation

For a neuron in the hidden layer:

z = Wx + b

a = f(z)

Where:

* x = input vector
* W = weight matrix
* b = bias vector
* z = pre-activation values
* f = activation function
* a = activated outputs

For a network with multiple layers, the calculations can be represented as:

a⁽⁰⁾ = x

a⁽ˡ⁾ = f⁽ˡ⁾(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)

Here, l represents the layer number.

The output of each layer becomes the input to the next layer.

---

## 4. Python Implementation From Scratch

This example implements a small network with two inputs, two hidden neurons, and one output neuron.

```python
import numpy as np

# Input features
x = np.array([2.0, 3.0])

# Hidden layer parameters
W1 = np.array([
    [0.5, 0.3],
    [0.2, 0.4]
])

b1 = np.array([0.1, 0.2])

# Output layer parameters
W2 = np.array([0.7, 0.6])
b2 = 0.1

# Hidden layer: weighted sum
z1 = np.dot(x, W1) + b1

# Hidden layer: ReLU activation
a1 = np.maximum(0, z1)

# Output layer: weighted sum
z2 = np.dot(a1, W2) + b2

# Output layer: sigmoid activation
prediction = 1 / (1 + np.exp(-z2))

print("Hidden layer output:", a1)
print("Output logit:", z2)
print("Final prediction:", prediction)
```

### Understanding the code

* `x` contains the two input features.
* `W1` contains the weights connecting the input features to the hidden neurons.
* `b1` contains the hidden neurons' biases.
* `np.dot(x, W1)` calculates the weighted sums for the hidden layer.
* `np.maximum(0, z1)` applies ReLU.
* `W2` and `b2` are the output neuron's weights and bias.
* Sigmoid converts the output logit into a value between 0 and 1.

These parameters are manually chosen for demonstration. This code performs a forward pass only; it does not train the network.

---

## 5. Forward Propagation Using TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential([
    Input(shape=(2,)),
    Dense(2, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Two example samples, each with two features
X = tf.constant([
    [2.0, 3.0],
    [1.0, 4.0]
])

predictions = model(X, training=False)

print(predictions.numpy())
```

Keras initializes the model parameters and calculates predictions when the model is called. The parameters here are randomly initialized, so these predictions are not meaningful until the model is trained.

Calling `model(X, training=False)` performs a forward pass without updating the model's parameters.

---

## 6. Forward Propagation vs Backpropagation

| Forward Propagation                     | Backpropagation                                        |
| --------------------------------------- | ------------------------------------------------------ |
| Calculates predictions                  | Calculates gradients                                   |
| Passes information from input to output | Propagates gradient information from the loss backward |
| Uses current weights and biases         | Calculates how parameters affect the loss              |
| Does not update parameters by itself    | Supplies gradients used by the optimizer               |

Both are essential to the usual neural network training process.

---

## 7. Common Mistakes

* Forgetting to add the bias.
* Applying an activation function at the wrong stage.
* Using incompatible matrix dimensions.
* Confusing the weighted sum with the activated output.
* Assuming forward propagation alone trains the model.
* Assuming randomly initialized predictions are meaningful.

---

## 8. Interview Questions and Answers

### Q1. What is forward propagation?

Forward propagation is the process of passing input data through the layers of a neural network to calculate its prediction.

### Q2. Does forward propagation update weights?

No. Forward propagation calculates the output using the current parameters. Parameter updates normally happen during training through an optimizer using gradients.

### Q3. What happens inside a neuron during forward propagation?

The neuron calculates a weighted sum of its inputs, adds a bias, and applies an activation function.

### Q4. Why is forward propagation important?

It produces the predictions required to calculate the loss and evaluate how well the network performs.

### Q5. What is the difference between a weighted sum and an activation output?

The weighted sum is calculated before the activation function. The activation output is the result after applying that function.

### Q6. Can a neural network perform forward propagation without training?

Yes. A network can generate predictions using initialized parameters, but its predictions may not be useful until it has been trained.

### Q7. What role does the hidden layer play?

Hidden layers transform input representations and learn patterns that can help the network make predictions.

### Q8. What happens when data passes through multiple hidden layers?

Each layer transforms the output of the previous layer. The final output is calculated after the data passes through the network's layers.

### Q9. How is forward propagation related to loss?

The prediction produced by forward propagation is compared with the actual target using a loss function. During training, this loss is used to calculate gradients for parameter updates.

### Q10. What is the difference between inference and training?

Inference uses a trained model to generate predictions. Training repeatedly calculates predictions, evaluates loss, computes gradients, and updates parameters.

---

## 9. Quick Revision

Forward propagation follows this sequence:

**Input → Weighted Sum → Bias → Activation → Next Layer → Final Prediction**

Remember:

* Every layer uses its own weights and biases.
* Activation functions transform the weighted sums.
* The final layer produces the model's output.
* Forward propagation alone does not update the parameters.

## 10. Practical Exercise

1. Run the NumPy implementation.
2. Change one weight and observe how the prediction changes.
3. Change the bias and compare the output.
4. Replace ReLU with tanh in the hidden layer.
5. Explain why changing a hidden-layer parameter can affect the final prediction.
