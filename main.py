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


X, Y = spiral_data(samples=100, classes=3)

print(X)

dense1 = Layer_Dense(2, 3)

dense1.forward(X)

print(dense1.output[:5])