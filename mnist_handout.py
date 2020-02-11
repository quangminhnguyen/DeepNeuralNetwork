from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

# Display the 150-th "5" digit from the training set
# imshow(M["train5"][6000].reshape((28,28)), cmap=cm.gray)
# show()
# 5422 images in train5


# 893 images in the test5
imshow(M["test1"][1134].reshape((28,28)), cmap=cm.gray)
show()

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    print("test {}".format(L0.shape))
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    print(sum(output))
    return L0, L1, output
    
def cross_entropy(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 


def output(W, x, b):
    L = dot(W.T, x) + b
    output = softmax(L)
    return output
    
#Load sample weights for the multilayer neural network
snapshot = cPickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

print("b0.shape = {}".format(b0.shape))
print("b1.shape = {}".format(b1.shape))
print("WO.shape = {}".format(W0.shape))
print("W1.shape = {}".format(W1.shape))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][0:1].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)
print(output.shape)
print(y)
################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
