import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist_load import load_mnist
from common.function import sigmoid, softmax
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import mnist

def get_data():
    #(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    return x_test, t_test


x, t = get_data()
print(x[0])
