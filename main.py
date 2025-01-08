# import numpy as np


# inputs = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
# inputs = np.array(inputs, dtype=np.float32)

# weights = [
#     [0.1, 2.3, -0.5, -0.2],
#     [1.3, -2.1, -2.2, 1.3],
#     [-3.1, 1.1, 0.2, -1.0],
# ]
# weights = np.array(weights, dtype=np.float32)
# bias = 3

# output = np.dot(inputs, weights.T) + bias

# print(output)

import numpy as np
# import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

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


X, Y = spiral_data(samples=100, classes=3)

print(X)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense1.forward(X)
activation1.forward(dense1.output)

print(dense1.output[:5])
print(activation1.output)