# Artificial Neural Network (ANN) — Fundamentals

## 1. What is an Artificial Neural Network?

An Artificial Neural Network (ANN) is a machine learning model inspired by the way biological neurons process information.

It learns patterns from data by adjusting weights and biases during training.

## 2. Why do we use ANN?

ANNs can learn complex relationships between input features and output values.

Common applications include:

* Classification
* Regression
* Pattern recognition
* Image and speech processing
* Medical prediction

## 3. Basic Architecture

An ANN typically consists of three types of layers:

1. Input Layer — receives input features.
2. Hidden Layer(s) — learns patterns and relationships.
3. Output Layer — generates the final prediction.

## 4. How does an ANN learn?

1. Input data enters the network.
2. Neurons calculate weighted sums and apply activation functions.
3. The network generates a prediction.
4. A loss function measures the prediction error.
5. Backpropagation calculates gradients.
6. An optimizer updates the weights and biases.

This process repeats over multiple training iterations.

## 5. Important Terms

* Neuron: A computational unit in the network.
* Weights: Learnable parameters that control input contributions.
* Bias: A learnable parameter that shifts the neuron's weighted sum.
* Activation Function: Introduces non-linearity.
* Loss Function: Measures prediction error.
* Optimizer: Updates model parameters to reduce the loss.
* Epoch: One complete pass through the training dataset.

## 6. Key Takeaway

An ANN learns by repeatedly making predictions, measuring errors, calculating gradients, and updating its parameters.

Understanding neurons, weights, biases, activation functions, forward propagation, and backpropagation is essential before implementing an ANN from scratch.

---

## 7. Interview Questions and Answers

### Q1. What is an Artificial Neural Network (ANN)?

**Answer:**

An Artificial Neural Network is a machine learning model inspired by biological neural networks. It consists of interconnected neurons that learn patterns from data by adjusting weights and biases during training.

### Q2. What are the main layers of an ANN?

**Answer:**

An ANN typically contains:

* **Input Layer:** Receives input features.
* **Hidden Layer(s):** Learns patterns and relationships from the input.
* **Output Layer:** Produces the final prediction.

### Q3. How does an ANN learn from data?

**Answer:**

An ANN learns through the following process:

1. Forward propagation generates predictions.
2. A loss function calculates the prediction error.
3. Backpropagation calculates gradients of the loss with respect to model parameters.
4. An optimizer updates weights and biases.
5. The process repeats until training finishes or a stopping condition is reached.

### Q4. What are weights and biases in an ANN?

**Answer:**

Weights determine the contribution of each input to a neuron's output, while biases allow the neuron to shift its activation independently of the weighted inputs.

Both are learnable parameters updated during training.

### Q5. What is an activation function, and why is it needed?

**Answer:**

An activation function transforms a neuron's weighted sum into its output. Non-linear activation functions allow a network to learn complex, non-linear relationships.

Common examples include ReLU, sigmoid, and tanh.

### Q6. What is the difference between an epoch, batch, and iteration?

**Answer:**

* **Epoch:** One complete pass through the training dataset.
* **Batch:** A subset of training samples processed together.
* **Iteration:** One parameter-update step, typically performed after processing a batch.

For example, if a dataset has 1,000 samples and the batch size is 100, one epoch typically contains 10 iterations when all batches are processed once.

### Q7. What is the difference between forward propagation and backpropagation?

**Answer:**

Forward propagation passes inputs through the network to generate predictions and calculate the loss.

Backpropagation calculates gradients of the loss with respect to the network's parameters using the chain rule. These gradients are then used by an optimizer to update the parameters.

### Q8. What is a loss function?

**Answer:**

A loss function measures how different a model's predictions are from the actual target values.

Examples:

* Binary cross-entropy for binary classification.
* Categorical cross-entropy for multiclass classification.
* Mean squared error (MSE) for regression.

### Q9. What is an optimizer in deep learning?

**Answer:**

An optimizer uses gradients to update model parameters and minimize the loss function.

Common optimizers include SGD, RMSprop, and Adam.

### Q10. What is overfitting in an ANN?

**Answer:**

Overfitting occurs when a model learns the training data too specifically, including noise, and performs poorly on unseen data.

It can be reduced using techniques such as dropout, regularization, early stopping, and appropriate data augmentation where applicable.

### Q11. What is the difference between ANN and traditional machine learning algorithms?

**Answer:**

Traditional machine learning algorithms often rely more on manually designed features, depending on the algorithm and task. ANNs learn representations through interconnected layers and can model complex relationships, but they may require more data and computational resources.

Neither approach is universally better; performance depends on the problem and dataset.

### Q12. What is the difference between shallow and deep neural networks?

**Answer:**

A shallow neural network has a relatively small number of hidden layers, while a deep neural network contains multiple hidden layers.

Deeper networks can learn hierarchical representations, although they can also be harder to train and tune.

### Q13. Can an ANN solve both classification and regression problems?

**Answer:**

Yes. ANNs can solve classification and regression tasks.

For example:

* Classification: Predicting whether a tumour is benign or malignant.
* Regression: Predicting a house's price.

The output layer, activation function, and loss function are selected according to the task.

### Q14. What happens if an ANN has too many hidden layers?

**Answer:**

Adding hidden layers increases model capacity, but it does not guarantee better performance. An unnecessarily complex network may require more computation, be harder to optimize, and overfit the training data.

The architecture should be selected based on validation performance and the problem's requirements.

### Q15. Why is data normalization or standardization useful for ANN training?

**Answer:**

Features with very different numerical scales can make optimization less effective. Normalization or standardization can improve training stability and convergence.

The preprocessing method should suit the dataset and model. Fit preprocessing transformations on training data only, then apply them to validation and test data to avoid data leakage.

---

## 8. Quick Revision

Before an ANN interview, make sure you can explain:

* What an ANN is and how it works.
* The roles of input, hidden, and output layers.
* Weights, biases, and activation functions.
* Forward propagation and backpropagation.
* Loss functions and optimizers.
* Epochs, batches, and iterations.
* Overfitting and methods to reduce it.
* Classification versus regression.
* The importance of feature scaling.

