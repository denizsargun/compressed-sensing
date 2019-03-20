import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import time

def getIm():
	# flatten = return grayscale
	# path leads to a sample mnist image of handwritten number "5"
	path = '/home/denizsargun/Documents/github/new_ideas/compressed_sensing/code/data_csv'
	return spimg.imread(path+'/mnist_sample.jpg', flatten=True, mode='L')

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def sampleIm(im):
	ny,nx = im.shape
	ratio = 0.1
	k = round(nx*ny*ratio)
	ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
	samples = im.T.flat[ri]
	samples = np.expand_dims(samples,axis=1)
	return samples.reshape((samples.shape[0],)), ri

def recover(samples,ri,shape):
	nx = shape[0]
	ny = shape[1]
	A = np.kron(spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0))
	A = A[ri,:]
	vx = cvx.Variable(nx*ny)
	objective = cvx.Minimize(cvx.norm(vx,1))
	# "inverse transform = samples" constraint
	constraints = [A*vx == samples]
	prob = cvx.Problem(objective,constraints)
	result = prob.solve(verbose=True)
	# transform solution
	recTran = np.array(vx.value).squeeze()
	recTran = recTran.reshape(nx,ny).T # stack columns
	return idct2(recTran)

im = getIm()
samples, ri = sampleIm(im)
recIm = recover(samples,ri,im.shape)
plt.imshow(im)
plt.figure()
plt.imshow(recIm)
plt.show()