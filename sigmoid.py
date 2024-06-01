import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X= np.array((3,-3))
print(sigmoid(X))
