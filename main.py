import numpy as np
# import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, vertical_data

nnfs.init()


class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



## Code for ReLU activation
# NB: f(x) = 0 if x<0 else x
class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


## Softmax activation
class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


class Loss:

    def calculate(self, output, y): # where why is the expected output

        sample_loss = self.forward(output, y)

        data_loss = np.mean(sample_loss)

        return data_loss


class Accuracy:

    def calculate(self, output, target):
        predictions = np.argmax(output, axis=1)
        class_targets = target

        if len(target.shape) == 2:
            class_targets = np.argmax(target, axis=1)
        
        return np.mean(class_targets==predictions)


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):

        # Number of samples per batch
        samples = len(y_pred)

        # Clipping data from both sides to reduce undefined
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum (y_pred_clipped * y_true, axis=1)

        # Loss
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood


X, Y = vertical_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()
accuracy_function = Accuracy()

best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_biases = dense2.biases.copy()

lowest_loss = 9999

for iteration in range(10000):
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, Y)

    if loss < lowest_loss:
        best_dense1_biases = dense1.biases.copy()
        best_dense1_weights = dense1.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        print(f"New set of weights and biases | acc: {accuracy_function.calculate(activation2.output, Y)} loss: {loss}")
        lowest_loss = loss

    


print(dense1.output[:5])
print(activation1.output)


# print(activation2.output[:20])

loss = loss_function.calculate(activation2.output, Y)

# print("loss:", loss)
# print("accuracy: ", accuracy_function.calculate(activation2.output, Y))