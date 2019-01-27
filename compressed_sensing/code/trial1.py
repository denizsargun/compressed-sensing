import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx

# n = 5000
# t = np.linspace(0, 1/8, n)
# y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
# yt = spfft.dct(y, norm='ortho')

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed
Xorig = spimg.imread('escher_waterfall.jpeg', flatten=True, mode='L') # read in grayscale
X = spimg.zoom(Xorig, 0.04)
ny,nx = X.shape