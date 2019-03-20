# https://www.python-course.eu/neural_network_mnist.php
import numpy as np
import matplotlib.pyplot as plt
import os

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
path = "/home/denizsargun/Downloads/mnist_data/csv"
fileTrain = os.path.join(path,"mnist_train.csv")
train_data = np.loadtxt(fileTrain,delimiter=",")
fileTest = os.path.join(path,"mnist_test.csv")
test_data = np.loadtxt(fileTest,delimiter=",")

# 0-255 to 0.01-0.99
train_imgs = np.asfarray(train_data[:,1:])/255*.98+.01
test_imgs = np.asfarray(test_data[:,1:])/255*.98+.01
train_labels = np.asfarray(train_data[:,0])
test_labels = np.asfarray(test_data[:,0])

# turn label into one-hot representation
lr = np.arange(10)
one_hot = (lr==label).astype(np.int)

# transform labels into one hot representation
# we don't want zeroes and ones in the labels neither
train_labels_one_hot = [(lr==train_label).astype(np.float)*.98+.01
						for train_label in train_labels]
test_labels_one_hot = [(lr==test_label).astype(np.float)*.98+.01
					   for test_label in test_labels]
