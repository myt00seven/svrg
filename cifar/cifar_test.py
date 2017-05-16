import cPickle as pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


data_x, data_y = load_CIFAR_batch("/home/yma/git/data/cifar-10-python.tar.gz")