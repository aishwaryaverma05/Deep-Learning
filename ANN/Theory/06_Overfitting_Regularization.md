# 06. Overfitting, Underfitting and Regularization in ANN

## 1. Introduction

When we train an Artificial Neural Network (ANN), our goal is to make it learn patterns from training data and perform well on unseen data.

However, sometimes a model learns too much from the training data or fails to learn enough. These problems are called **overfitting** and **underfitting**.

Regularization techniques help us improve the model's ability to generalize to new data.

---

## 2. What Is Overfitting?

**Overfitting occurs when a model learns the training data too closely, including noise and unnecessary details.**

An overfitted model performs very well on training data but poorly on unseen data.

### Example

Suppose we train an ANN to classify breast tumors as benign or malignant.

* Training accuracy: 99%
* Validation accuracy: 78%

The model performs extremely well on training data but struggles with new examples. This may indicate overfitting.

### Causes of Overfitting

* Model is too complex.
* Too many parameters relative to the available data.
* Training for too many epochs.
* Insufficient training data.
* No regularization.

### How to Reduce Overfitting

* Apply dropout.
* Use L1 or L2 regularization.
* Apply early stopping.
* Collect more training data or use suitable data augmentation.
* Reduce model complexity.

---

## 3. What Is Underfitting?

**Underfitting occurs when a model is too simple or has not learned the underlying patterns in the data.**

An underfitted model performs poorly on both training and validation data.

### Example

* Training accuracy: 65%
* Validation accuracy: 63%

The model is not learning the important patterns properly.

### Causes of Underfitting

* Model is too simple.
* Too few training epochs.
* Insufficient useful features.
* Excessive regularization.
* Poor model configuration.

### How to Reduce Underfitting

* Increase model complexity when appropriate.
* Train for more epochs.
* Improve feature selection or preprocessing.
* Reduce excessive regularization.
* Tune the model's hyperparameters.

---

## 4. Overfitting vs Underfitting

| Feature                | Overfitting                             | Underfitting                      |
| ---------------------- | --------------------------------------- | --------------------------------- |
| Training performance   | Very good                               | Poor                              |
| Validation performance | Significantly worse than training       | Poor                              |
| Main problem           | Learns noise and specific details       | Fails to learn important patterns |
| Model complexity       | Often too high                          | Often too low                     |
| Possible solution      | Regularization, dropout, early stopping | More capacity or better training  |

A good model learns useful patterns and generalizes well to unseen data.

---

## 5. Training, Validation and Testing Data

Dataset splitting helps us evaluate whether our model generalizes beyond the data used to train it.

### Training Data

Used to learn model parameters, such as weights and biases.

### Validation Data

Used to evaluate the model during development and tune hyperparameters.

### Testing Data

Used for the final evaluation after model development and tuning are complete.

### Example Split

A possible split is:

* Training: 70%
* Validation: 15%
* Testing: 15%

These proportions are not fixed. The best split depends on the dataset size and problem.

**Important:** Do not repeatedly tune your model based on test-set results. Doing so can indirectly overfit your model-selection process to the test set.

---

## 6. What Is Regularization?

**Regularization is a technique used to reduce overfitting by discouraging unnecessarily complex models.**

It adds a penalty to the original loss function.

General formula:

$$
L_{\text{total}} = L_{\text{original}} + L_{\text{penalty}}
$$

The model tries to minimize the original prediction error while also controlling complexity.

Two common regularization techniques are L1 and L2 regularization.

---

## 7. L1 Regularization

L1 regularization adds the absolute values of model weights to the loss function.

$$
L_{\text{total}} = L_{\text{original}} + \lambda \sum_i |w_i|
$$

Where:

* \(L_{\text{original}}\): original loss.
* \(w_i\): model weights.
* \(\lambda\): regularization strength.

### How Does It Work?

L1 regularization penalizes large weights and can drive some weights exactly to zero.

This can produce a sparse model, where some features or connections contribute less or become effectively unused.

### TensorFlow Example

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L1

model = Sequential([
    Dense(
        64,
        activation="relu",
        kernel_regularizer=L1(0.001),
        input_shape=(30,)
    ),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

Here, `L1(0.001)` applies an L1 penalty to the weights of the first Dense layer.

---

## 8. L2 Regularization

L2 regularization adds the squared values of model weights to the loss function.

$$
L_{\text{total}} = L_{\text{original}} + \lambda \sum_i w_i^2
$$

### How Does It Work?

L2 regularization discourages excessively large weights by penalizing their squared magnitudes.

It generally encourages weights to become smaller rather than forcing them exactly to zero.

L2 regularization is also commonly called **weight decay**, although the exact implementation of weight decay can differ from adding an L2 penalty to the loss.

### TensorFlow Example

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2

model = Sequential([
    Dense(
        64,
        activation="relu",
        kernel_regularizer=L2(0.001),
        input_shape=(30,)
    ),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
```

### L1 vs L2

| L1 Regularization                          | L2 Regularization                       |
| ------------------------------------------ | --------------------------------------- |
| Uses absolute weights                      | Uses squared weights                    |
| Can produce zero weights                   | Usually shrinks weights toward zero     |
| Can encourage sparsity                     | Encourages smaller, distributed weights |
| Useful when sparse solutions are desirable | Commonly used to control large weights  |

The regularization coefficient must be tuned. Too much regularization can cause underfitting.

---

## 9. Dropout

**Dropout is a regularization technique that randomly disables a fraction of neurons during training.**

For example, a dropout rate of `0.3` means approximately 30% of the eligible layer's outputs are randomly dropped during each training step.

### Why Do We Use Dropout?

Without dropout, neurons may become overly dependent on specific other neurons.

Dropout encourages the network to learn more robust patterns.

### TensorFlow Example

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation="relu", input_shape=(30,)),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])
```

### Important Points

* Dropout is active during training.
* It is automatically disabled during normal inference and evaluation.
* Dropout rates such as `0.2` or `0.3` are starting points, not universal best values.
* Excessive dropout may cause underfitting.

---

## 10. Early Stopping

**Early stopping stops training when the model stops improving on a monitored validation metric.**

Training for too many epochs may cause the model to overfit.

Early stopping helps prevent unnecessary training.

### TensorFlow Example

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)
```

### Explanation

* `monitor="val_loss"`: monitor validation loss.
* `patience=5`: wait up to five consecutive epochs without improvement before stopping.
* `restore_best_weights=True`: restore the weights from the epoch with the best monitored validation loss.

Early stopping does not necessarily stop at a fixed epoch. It depends on validation performance.

---

## 11. Practical Example: Detecting Overfitting

Consider the following results:

| Epoch | Training Accuracy | Validation Accuracy |
| ----- | ----------------: | ------------------: |
| 1     |               70% |                 68% |
| 5     |               85% |                 82% |
| 10    |               94% |                 88% |
| 20    |               99% |                 81% |

At epoch 10, the model performs well on both training and validation data.

By epoch 20, training accuracy continues increasing, but validation accuracy decreases.

This suggests that the model may be overfitting. Early stopping based on validation loss could help select a better checkpoint.

**Remember:** Accuracy alone does not tell the entire story. Also examine validation loss and, for classification problems, suitable metrics such as precision, recall, F1-score and ROC-AUC.

---

## 12. Complete TensorFlow Example

This example demonstrates regularization and early stopping for binary classification.

Assumptions:

* `X_train` and `y_train` contain training data.
* `X_val` and `y_val` contain validation data.
* Each input sample has 30 numerical features.
* Features have been appropriately preprocessed.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(
        64,
        activation="relu",
        kernel_regularizer=L2(0.001),
        input_shape=(30,)
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu",
        kernel_regularizer=L2(0.001)
    ),

    Dropout(0.2),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
```

Here, `X_test` and `y_test` must be kept separate from training and validation.

This code is a template; actual performance depends on the dataset, preprocessing, model configuration and train/validation split.

---

## 13. Interview Questions and Answers

### Q1. What is overfitting?

Overfitting occurs when a model learns the training data too closely, including noise, and performs poorly on unseen data.

### Q2. What is underfitting?

Underfitting occurs when a model fails to learn the important patterns in the data and performs poorly on both training and validation data.

### Q3. What is regularization?

Regularization discourages unnecessarily complex models by adding a penalty or constraint during training, helping reduce overfitting.

### Q4. What is the difference between L1 and L2 regularization?

L1 penalizes the absolute values of weights and can produce zero weights. L2 penalizes squared weights and generally shrinks weights toward zero.

### Q5. What is dropout in ANN?

Dropout randomly disables a fraction of neurons during training to reduce co-adaptation and improve generalization.

### Q6. Is dropout active during testing?

No. In standard Keras models, dropout is automatically disabled during inference and evaluation.

### Q7. What is early stopping?

Early stopping terminates training when a monitored validation metric stops improving for a specified patience period.

### Q8. Why do we need validation data?

Validation data helps compare models and tune hyperparameters without using the test set for repeated decisions.

### Q9. Can a model have high training accuracy and low validation accuracy?

Yes. This is a common sign of overfitting, although the data split and evaluation process should also be checked.

### Q10. Can regularization cause underfitting?

Yes. Excessive regularization can restrict learning too much and cause poor performance on both training and validation data.

### Q11. What is generalization in machine learning?

Generalization is the ability of a trained model to make useful predictions on unseen data drawn from the relevant target distribution.

### Q12. Which technique should we use to reduce overfitting?

There is no single best technique for every model. Options include dropout, L1/L2 regularization, early stopping, additional representative data and reducing model complexity.

---

## 14. Key Takeaways

* Overfitting: learns training data too closely.
* Underfitting: fails to learn important patterns.
* Training data: used to learn model parameters.
* Validation data: used for model selection and tuning.
* Test data: used for final evaluation.
* L1: can encourage sparse weights.
* L2: discourages excessively large weights.
* Dropout: randomly disables neurons during training.
* Early stopping: stops training when validation performance stops improving.
* Generalization: the ultimate goal is good performance on unseen data.

**Next Topic:** `07_ANN_Model_Evaluation_Metrics.md`

We will learn confusion matrix, accuracy, precision, recall, F1-score, ROC-AUC and how to evaluate a classification model.
