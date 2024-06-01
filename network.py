import numpy as np
from sigmoid import sigmoid

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(netwrork,x):
    W1, W2, W3 = netwrork['W1'], netwrork['W2'], netwrork['W3']
    b1, b2, b3 = netwrork['b1'], netwrork['b2'], netwrork['b3']

    a1= np.dot(x, W1) + b1
    print("ai: ", a1)
    z1 = sigmoid(a1)

    return z1

y = forward(init_network(), np.array([1.0, 0.5]))
print(y)
#print(init_network())