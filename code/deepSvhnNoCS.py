# read+learn street view house numbers dataset
# use fully connected feedforward neural net

import sys
import os
import inspect
import scipy.io
import numpy as np
from neuralNetwork import * # main class imported is "NeuralNetwork"

# location of the data
path = "/home/denizsargun/Downloads/svhn_data/"
# running on Ohio supercomputer center (OSC):
# filename = inspect.getframeinfo(inspect.currentframe()).filename
# path = os.path.dirname(os.path.abspath(filename))
trainFile = "train_32x32.mat"
testFile = "test_32x32.mat"

# read data
trainData = scipy.io.loadmat(os.path.join(path,trainFile))
testData = scipy.io.loadmat(os.path.join(path,testFile))

# how to access 0th image
# trainData['X'][:,:,:,0]

# shift labels 1-10 to 0-9
trainData['y'] = np.ravel(np.remainder(trainData['y'],10))
testData['y'] = np.ravel(np.remainder(testData['y'],10))

# data specifications
trainSize = trainData['y'].shape[0]
# trainSize = 10000
testSize = testData['y'].shape[0]
# testSize = 10000
imageSize = 32*32*3

# create 1D version of 3D images
squeezeTrain = np.zeros((imageSize,trainSize))
squeezeTest = np.zeros((imageSize,testSize))

for i in range(trainSize):
	squeezeTrain[:,i] = trainData['X'][:,:,:,i].reshape(-1)

for i in range(testSize):
	squeezeTest[:,i] = testData['X'][:,:,:,i].reshape(-1)

# turn training labels into list of one-hot arrays
eye = np.eye(10) # there are 10 classes
trainLabels = [eye[trainData['y'][i]] for i in range(trainSize)]

# keep test labels
testLabels = testData['y']

# replace 4D arrays with 2D
trainData = squeezeTrain
testData = squeezeTest

# create, train and evaluate feedforward fully connected NN with backpropagation and SGD
NN = NeuralNetwork(network_structure=[3072, 80, 80, 10],
                                learning_rate=0.01,
                                bias=0.5)
NN.train(trainData.transpose(),trainLabels,epochs=10)
NN.evaluate(testData.transpose(),testLabels) # evaluate using indexed labels instead of one-hot labels

# running on OSC
# sys.exit(0)