# Implementation of a Single Layer Perceptron as a binary classifier

The objective of this code is to implement a Single Layer Perceptron (SLP) for binary classification. Specifically, the code demonstrates the perceptron learning algorithm by solving the XOR problem, where the task is to predict binary outputs (0 or 1) based on two binary input values. The code initializes the perceptron, trains it on the XOR dataset, and tests its predictions to evaluate its performance.

- The perceptron will be trained using a simple dataset.
- The weights and bias will be updated using the Perceptron Learning Rule.
- The activation function will be the step function (as it's simpler and more appropriate for binary classification).
- The dataset will have binary labels (0 or 1).

---

### Explanation of the Code
**1. Class Definition:**
The SingleLayerPerceptron class contains methods to initialize the perceptron, apply an activation function, predict outputs, and train the model.

**2. Initialization:**
In the __init__ method, the weights are initialized to zero, and the bias is set to zero. The learning rate and the number of epochs are also specified for training.

**3. Activation Function:**
The activation_function method implements a step function that outputs 1 if the input is greater than or equal to zero and 0 otherwise. This function is suitable for binary classification tasks.

**4. Prediction:**
The predict method calculates the weighted sum of the inputs, adds the bias, and applies the activation function to generate the output.

**5. Training:**
The train method uses the Perceptron Learning Rule to update the weights and bias based on the difference between the predicted and actual labels. This is done for a specified number of epochs.

**6. Example Usage:**
The main block defines the training dataset for the XOR problem and the corresponding binary labels. It creates an instance of the SingleLayerPerceptron, trains it on the dataset, and tests it by printing the predicted outputs for each input

---

### **Algorithm: Single Layer Perceptron for Binary Classification**

1. **Initialize parameters**:
   - Initialize weights `w` randomly or as zeros for each input feature.
   - Initialize bias `b` as zero or a small random value.
   - Set the learning rate `η` and the number of epochs.

2. **Training process**:
   - For each epoch (repeat for a fixed number of iterations):
     1. **For each input data point** in the dataset:
        - Compute the weighted sum `z = w.x + b`.
        - Apply the activation function:
          - If `z ≥ 0`, set output `y_pred = 1`.
          - If `z < 0`, set output `y_pred = 0`.
        - Compare the predicted output `y_pred` with the actual label `y_true`.
        - Update the weights and bias:
          - `w = w + η * (y_true - y_pred) * x`
          - `b = b + η * (y_true - y_pred)`

3. **Prediction**:
   - After training, use the updated weights `w` and bias `b` to predict outputs for new input data:
     - Calculate `z = w.x + b`.
     - Apply the activation function:
       - Output `1` if `z ≥ 0`, otherwise output `0`.

4. **Stopping condition**:
   - Stop training after a fixed number of epochs or when the model achieves high accuracy on the training dataset.


This algorithm provides a simple process for implementing a **Single Layer Perceptron (SLP)** for binary classification problems such as **XOR logic gates** or **spam detection**.

---

### Summary
The code implements a Single Layer Perceptron (SLP) to solve a binary classification problem using the XOR logic gate as an example. The perceptron is trained using input values and corresponding labels, where the activation function determines whether the output is 0 or 1.
By adjusting the weights and bias through the Perceptron Learning Rule, the model learns to predict the correct output.

This code can also be adapted for real-world applications like spam email detection, where the perceptron classifies emails based on input features, such as the presence of specific words or patterns.








