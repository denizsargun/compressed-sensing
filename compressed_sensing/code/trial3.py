# reading MNIST

import _pickle
import gzip
import numpy
import os

path = '/home/denizsargun/Downloads/mnist_data/gzip'
file = os.path.join(path,'mnist.pkl.gz')
f = gzip.open(file, 'rb')
train_set, valid_set, test_set = _pickle.load(f,encoding='latin1')