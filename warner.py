import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

X, Y = vertical_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap='brg')
plt.show()