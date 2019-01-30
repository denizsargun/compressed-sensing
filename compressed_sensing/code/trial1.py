# http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import time

start = time.time()

# n = 5000
# t = np.linspace(0, 1/8, n)
# y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
# yt = spfft.dct(y, norm='ortho')

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed
# Xorig = spimg.imread('escher_waterfall.jpg', flatten=True, mode='L') # read in grayscale
path = '/home/denizsargun/Documents/github/new_ideas/compressed_sensing/code/data_csv'
X = spimg.imread(path+'/mnist_sample.jpg', flatten=True, mode='L')
# plt.imshow(Xorig)
plt.imshow(X)
plt.show()

# X = spimg.zoom(Xorig, 0.04)
# plt.figure()
# plt.imshow(X)
# plt.show()

ny,nx = X.shape
k = round(nx * ny * 0.5) # 50% sample
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
b = X.T.flat[ri]
b = np.expand_dims(b, axis=1)
b = b.reshape((b.shape[0],))

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
A = A[ri,:] # same as phi times kron
end = time.time()
print(end - start)

# do L1 optimizationc
vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == b]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

# reconstruct signal
Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)

# confirm solution
if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    print('Warning: values at sample indices don\'t match original.')

# create images of mask (for visualization)
mask = np.zeros(X.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]
# plt.imshow(Xat2)
plt.imshow(Xm)
plt.show()
end = time.time()
print(end - start)