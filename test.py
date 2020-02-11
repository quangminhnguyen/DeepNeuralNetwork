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
import random

import cPickle

import os
from scipy.io import loadmat


def part1():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")
    f, axarr = plt.subplots(10, 10)

    for i in range(10):
        train_num = "train" + str(i)
        test_num = "test" + str(i)
        train_size = M[train_num].shape[0]
        test_size = M[test_num].shape[0]
        print(train_size)
        random.seed(i)
        rand_train_img = random.sample(range(0, train_size), 8)
        rand_test_img = random.sample(range(0, test_size), 2)
        
        # print(i)
        # print(rand_train_img)
        # print(rand_test_img)
        
        # Display 8 training images and 2 test images.
        for k in range(len(rand_train_img)):
            img = M[train_num][k].reshape((28,28))
            axarr[i, k].imshow(img, cmap=cm.gray)
            axarr[i, k].axis('off')
        
        
        for k in range(len(rand_test_img)):
            img = M[train_num][k + 8].reshape((28,28))
            axarr[i, k + 8].imshow(img, cmap=cm.gray)
            axarr[i, k + 8].axis('off')
        
    plt.show()
    
    
    
    
def test_part4():
    x = np.array([1, 2, 3, 4, 5, 6])
    y1 = np.array([1, 2, 3, 4, 5, 6])
    y2 = np.array([6, 5, 4, 3, 5, 6])
    plt.subplot(2,1,1)
    plt.plot(y1, color='red', lw = 2, label="helo")
    plt.plot(y2, color='blue', lw = 2, label="hoho")
    plt.plot(y2, color='pink', lw = 2, label="tete")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
    plt.ylabel('Number of occurrences')
    #pyplot.xlabel('In degree') 
    plt.xlabel('Out degree')
    
    plt.show()

test_part4()


        

        
    
    
    
    
    
    
        
    


