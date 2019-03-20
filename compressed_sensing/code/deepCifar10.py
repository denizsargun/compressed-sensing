# cifar-10 data set
# https://www.cs.toronto.edu/~kriz/cifar.html
# feedforward fully connected deep neural net

import os
import pickle
import numpy as np
from neuralNetwork import * # main class imported is "NeuralNetwork"

# read data
path = "/home/denizsargun/Downloads/cifar-10_data/"
trainFileHeader = "data_batch_" # there are 5 training batches, data_batch_1,...,data_batch_5
testFile = "test_batch"

trainData = {'data':np.zeros((50000,3072)),'labels':[]}
for i in range(1,6): # i = 1,...,5
	trainFile = os.path.join(path,trainFileHeader+str(i))
	with open(trainFile,'rb') as dum:
		dumDict = pickle.load(dum,encoding='bytes')
	trainData['data'][10000*(i-1):10000*i,:] = dumDict[b'data']
	trainData['labels'] = trainData['labels']+dumDict[b'labels']

testData = {}
with open(os.path.join(path,testFile),'rb') as dum:
	testDict = pickle.load(dum,encoding='bytes')

testData['data'] = testDict[b'data']
testData['labels'] = testDict[b'labels']

# data specs
trainSize = 50000
testSize = 10000
imageSize = 32*32*3

# turn training labels into list of one-hot arrays
eye = np.eye(10) # there are 10 classes
trainLabels = [eye[trainData['labels'][i]] for i in range(trainSize)]

# create, train and evaluate feedforward fully connected NN with backpropagation and SGD
NN = NeuralNetwork(network_structure=[3072, 80, 80, 10],
                                learning_rate=0.01,
                                bias=0.5)
NN.train(trainData['data'],trainLabels,epochs=10)
NN.evaluate(testData['data'],testData['labels']) # evaluate using indexed labels instead of one-hot labels