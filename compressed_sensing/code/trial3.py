# reading MNIST

import _pickle
import gzip
import numpy
import os

path = '/home/denizsargun/Documents/github/new_ideas/compressed_sensing/code/DeepLearningTutorials/data'
file = os.path.join(path,'mnist.pkl.gz')
f = gzip.open(file, 'rb')
train_set, valid_set, test_set = _pickle.load(f,encoding='latin1')
