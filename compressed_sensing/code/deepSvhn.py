# read+learn street view house numbers dataset
# use fully connected feedforward neural net
# sample compressively using compressed sensing matrices

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
# trainSize = trainData['y'].shape[0]
trainSize = 20000
# testSize = testData['y'].shape[0]
testSize = 1000
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

# random sample training and test data
k = 100 # compressed sensing parameter
print('k = ',k)
trainRand = np.zeros((trainSize,k))
testRand = np.zeros((testSize,k))

# generate sampling vectors uniformly over the sphere with dimension image-pixels
# use standard Gaussian random vectors of size image-pixels and normalize their L2 norm
N = np.zeros((imageSize,k))
for i in np.arange(k):
	v = np.random.normal(size=imageSize)
	N[:,i] = v/np.linalg.norm(v)

# sample
for i in np.arange(trainSize):
	for j in np.arange(k):
		trainRand[i,j] = np.inner(trainData[:,i],N[:,j])
		if not i%1000 and j == 0:
			print('measuring ',i,'th train data')

for i in np.arange(testSize):
	for j in np.arange(k):
		testRand[i,j] = np.inner(testData[:,i],N[:,j])
		if not i%1000 and j == 0:
			print('measuring ',i,'th test data')

# create, train and evaluate feedforward fully connected NN with backpropagation and SGD
NN = NeuralNetwork(network_structure=[k, 80, 80, 10],
                                learning_rate=0.01,
                                bias=0.5)
NN.train(trainRand,trainLabels,epochs=10)
NN.evaluate(testRand,testLabels) # evaluate using indexed labels instead of one-hot labels

# running on OSC
# sys.exit(0)