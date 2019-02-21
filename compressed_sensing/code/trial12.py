# read+learn street view house numbers dataset
# use fully connected feedforward neural net
# sample compressively using compressed sensing matrices

import scipy.io

folder = "/home/denizsargun/Downloads/svhn_data/"
trainFile = "train_32x32.mat"
testFile = "test_32x32.mat"

trainData = scipy.io.loadmat(folder+trainFile)
testData = scipy.io.loadmat(folder+testFile)