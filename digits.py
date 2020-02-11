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
import shutil

import cPickle
import scipy.stats
import os
from scipy.io import loadmat

from auto_pick import *
#import tensorflow as tf


# ----------------------------------------------------------------
# PART 1 code
#-----------------------------------------------------------------
def part1():
    M = loadmat("mnist_all.mat")
    f, axarr = plt.subplots(10, 10)
    count_train_image = 0;
    count_test_image = 0;
    
    
    for i in range(10):
        train_num = "train" + str(i)
        test_num = "test" + str(i)
        train_size = M[train_num].shape[0]
        test_size = M[test_num].shape[0]
        count_train_image = count_train_image + train_size 
        count_test_image = count_test_image + test_size
        
        
        print("Train set of number {} has {} images.".format(i, train_size))
        print("Test set of number {} has {} images.".format(i, test_size))

        random.seed(i)
        rand_train_img = random.sample(range(0, train_size), 8)
        rand_test_img = random.sample(range(0, test_size), 2)
        
        
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
    
    print("In total there are {} trainning and {} testing images".format(count_train_image, count_test_image))

#part1()



# ----------------------------------------------------------------
# PART 2
#-----------------------------------------------------------------
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    

def output(W, x):
    # Add 1 on top of x, this is for the bias b. 
    x = vstack((ones((1, x.shape[1])), x))
    o = dot(W.T, x)
    p = softmax(o)
    return o, p
    



# ----------------------------------------------------------------
# PART 3
#-----------------------------------------------------------------

# Same as the function output(W,x) in part 2, but the bias is treated in another function.
def compute_p_part3(W,x):
    o = dot(W.T, x)
    p = softmax(o)
    return o, p


# Vectorized gradient function.
def vectorized_gradient_part3(x, y, W):
    x = vstack((ones((1, x.shape[1])), x))
    return dot(x, (compute_p_part3(W, x)[1] - y).T)
    

# Deriative with respect to w_ij
def finite_difference_gradient_part3(x, y, W, i , j):
    x = vstack((ones((1, x.shape[1])), x))
    sum = 0;
    for t in range(x.shape[1]):
        # sum = sum + x[i][t] * (dot(W.T[j], x[:,t]) - y[j][t])
        sum = sum + x[i][t] * (softmax(dot(W.T, x))[j][t] - y[j][t])
    return sum;
        
        
def part3():
    # say each images has 5 pixels (not include 1)
    # 2 images [1,1, 1, 1,5] ,[2, 1, 1, 1, 1]
    # 3 possible labels [1, 0, 0], [0, 1, 0], [0, 0, 1]
    # x = (5 + 1) x 2 matrix
    # y = 3 x 2 matrix
    # W.T = 3 x (5 + 1) matrix
    
    # Form an x
    img1 = array([1, 1, 1, 1, 5])
    img2 = array([2, 1, 1, 1, 1])
    img = vstack([img1, img2])
    x = img.T # x is in n * m, where n is number of pixels and m is number of samples.
    print("------------- x is (without 1 on top yet) ---------------")
    print(x)
    
    # Forms a W
    W = zeros((6, 3))
    print("----------- W is -------------")
    print(W) # W is in n * k, where n is number of pixels and k is number of labels.
    
    # Forms an y
    y = array([[1, 0, 0], [0, 0, 1]]);
    y = y.T; # y is in k * m, where k is number of labels and m is the sample size.
    
    print("---------vectorized gradient-----------")
    print(vectorized_gradient_part3(x, y, W))
    
    
    print("------gradient computed using finite diffeence------")
    g = zeros_like(vectorized_gradient_part3(x, y, W))
    # Compute every component of the gradient using the the finite difference method.
    for i in range(x.shape[0] + 1): # number of pixels + bias.
        for j in range(y.shape[0]): # number of labels.
            g[i, j] = finite_difference_gradient_part3(x, y, W, i, j)
    print(g)

# part3()




# ----------------------------------------------------------------
# PART 4
#-----------------------------------------------------------------

# Get the correctness rate of the training and test data, given the edge weights.
# x = n * m = number of pixels * sample size
# y = k * m = number of possible labels * sample size.
def get_correctness_rate(x, y, W):
    # result = dot(theta[1:].T, imgtest) + theta[0].T
    o = (dot(W[1:].T, x).T + W[0]).T
    p = softmax(o) 
    # p = k * m matrix, k = number of possible labels, m is number of labels.
    # y is also k * m matrix
    
    # turns both into m * k and se how many are matched.
    p = p.T 
    y = y.T
    
    count = 0;
    for i in range(p.shape[0]):
        if argmax(p[i]) == argmax(y[i]):
            count = count + 1;
    
    return float(count)/float(p.shape[0]) * 100
    
# visualized the weights of each of the outputs. 
def display_images(t):
    t = t[1:].T
    for i in range(t.shape[0]):
        file_name = "part4_img_" + str(i)
        img = t[i].reshape(28,28);
        
        # save the images into local directory.
        imsave(file_name, img)
    

# Runs gradient descent to find best weight
def grad_descent_part4(train, y_of_train, init_t, alpha, test, y_of_test):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 3500
    iter  = 0
    
    train_performance = []
    test_performance = []
    
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * vectorized_gradient_part3(train, y_of_train, t)
        # print(norm(t - prev_t))
        
        # Gets the correctness rate.
        train_correctness_rate = get_correctness_rate(train, y_of_train, t)
        test_correctness_rate = get_correctness_rate(test, y_of_test, t)
        train_performance.append(train_correctness_rate)
        test_performance.append(test_correctness_rate)
        
        if iter % 200 == 0:
            print "Itertions #", iter
            print("Performance:", train_correctness_rate, test_correctness_rate)        
        iter += 1
    
    print(t.size)
    
    # Save the 10 images of the 10 set of weights. #
    display_images(t)
    return (train_performance, test_performance)


# Plot the learning curve
def plot_graph(result):
    train = np.asarray(result[0])
    test = np.asarray(result[1])
    #print(train.shape)
    #print(test.shape)
    plt.subplot(2,1,1)
    plt.plot(train, '-', color='red', lw = 2)
    plt.plot(test, ':', color='green', lw = 2)
    plt.ylabel('Correctness rate (%)')
    plt.xlabel('Number of iterations')
    plt.title('Learning curves.')
    plt.show()


def part4():
    n = 785 # number of pixels + 1
    k = 10 # number of possible labels.
    m = 6000 # number of training images.
    
    # y = zeros((k,m)); # k is the number of possible labels.
    x = array([]); # training data
    y_of_x = array([]); # labels_train_data
    t = array([]); # test data
    y_of_t = array([]); # labels_test_data
    test_set = array([]);
    M = loadmat("mnist_all.mat")
    
    # Learns from all training samples, and test using 
    # all given training and testing samples.
    for i in range(k):
        train_num = "train" + str(i)
        test_num = "test" + str(i)
        
        print(i)
        train_size = M[train_num].shape[0]
        for j in range(train_size):
            img = M[train_num][j]
            
            if x.size == 0:
                x = vstack([img]);
            elif x.size > 0:
                x = vstack([x, img]);
            
            buffer_y = zeros(k);
            buffer_y[i] = 1;
            if y_of_x.size == 0:
                y_of_x = vstack([buffer_y])
            elif y_of_x.size > 0:
                y_of_x = vstack([y_of_x, buffer_y])
        
        
        test_size = M[test_num].shape[0]
        for j in range(test_size):
            img = M[test_num][j]
            
            if t.size == 0:
                t = vstack([img]);
            elif t.size > 0:
                t = vstack([t, img])
                
                
            buffer_y = zeros(k);
            buffer_y[i] = 1;
            if y_of_t.size == 0:
                y_of_t = vstack([buffer_y])
            elif y_of_t.size > 0:
                y_of_t = vstack([y_of_t, buffer_y])
        
    

    x = x.T # data for training.
    y_of_x = y_of_x.T # labels for training data.
    t = t.T # data for testing.
    y_of_t = y_of_t.T # labels for testing data.
    
    alpha = 0.000000000001;
    W0 = zeros((n, k));
    result = grad_descent_part4(x, y_of_x, W0, alpha, t, y_of_t);
    

    # Plot the learning curve. 
    plot_graph(result);
    print(y_of_t.shape)
    print(y_of_x.shape)
    print(x.shape)
    print(t.shape)
    
#part4();
    



# ----------------------------------------------------------------
# PART 5
#-----------------------------------------------------------------
# Linear regression Vectorized gradient function.
def vectorized_gradient_linear_part5(x, y, theta):
    x = vstack((ones_like(x), x))
    return dot(dot(2,x), (dot(theta.T, x) - y).T)


# Linear regression for gradient descent.
def grad_descent_linear_part5(x, y, init_t, alpha):
    EPS = 1e-10   #EPS = 10**(-10)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 50000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t = t - alpha * vectorized_gradient_linear_part5(x, y, t)
        if iter % 10000 == 0:
            print "Iteration #", iter
        iter += 1
    return t

# Neural network Vectorized gradient function.
def vectorized_gradient_part5(x, y, W):
    x = vstack((ones_like(x), x))
    return dot(x, (compute_p_part3(W, x)[1] - y).T)
    
# Neural network gradient descent. 
def grad_descent_logistic_part5(train, y_of_train, init_t, alpha):
    EPS = 1e-10   #EPS = 10**(-10)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 50000
    iter  = 0
        
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t =  t - alpha * vectorized_gradient_part5(train, y_of_train, t)
        if iter % 10000 == 0:
            print "Itertions #", iter

        iter += 1
        
    return t
    

    
def part5():
    m = 200
    n = 2 # number of feature + 1
    k = 2 # number of possible labels
    
    # Training data
    np.random.seed(1)
    mu, sigma = 0, 0.1 # mean and standard deviation.
    s = np.random.normal(mu, sigma, m)
    s = s.reshape((1,m))
        
    print("------- Training data with no outlier -------")
    print(s)
            
    # classification
    y = zeros((k, m))
    for i in range(m):
        if s[0][i] > 0:
            y[1][i]= 1;
        elif s[0][i] <= 0:
            y[0][i] = 1;
    
    # Let put an outlier into x.
    s[0][0] = 50000;
    
    print("------- Training data with an outlier added -------")
    print(s)        
    
    init_theta = np.zeros((n, k)) # each sample has only 1 feature + 1 for the bias.
    alpha = 0.000000000001
    
    # Linear gradient descent.
    theta = grad_descent_linear_part5(s, y, init_theta, alpha)
    # Logistic graident descent.
    weight = grad_descent_logistic_part5(s, y, init_theta, alpha)
    
    # Linear regression result on training data.
    linear_result = (dot(theta[1:].T, s) + theta[0].reshape(k,1))
    
    # Logistic regression on training data
    logistic_result = (dot(weight[1:].T, s) + weight[0].reshape(k,1))
    
    correct_count_linear = 0;
    corect_count_logistic = 0;
    
    y = y.T
    linear_result = linear_result.T
    logistic_result = logistic_result.T
    
    for i in range(m):
        if (argmax(y[i]) == argmax(linear_result[i])):
            correct_count_linear = correct_count_linear + 1;
        if (argmax(y[i]) == argmax(logistic_result[i])):
            corect_count_logistic = corect_count_logistic + 1;
    
    print("------- Performance report on training data -------")
    print("Linear regression on training set with the outlier.", float(correct_count_linear)/float(len(y)) * 100)
    print("Logistic regression on training set with the outlier.", float(corect_count_logistic)/float(len(y)) * 100)
    
    
    
    # test data
    np.random.seed(3)
    
    test_data = np.random.normal(mu, sigma, m)
    test_data = test_data.reshape((1, m))
    
    print("------- Test data with no outlier. -------")
    print(test_data)  
    
    # Classification of the test data.
    y_test = zeros((k, m))
    for i in range(m):
        if test_data[0][i] > 0:
            y_test[1][i]= 1;
        elif test_data[0][i] <= 0:
            y_test[0][i] = 1;
            
    linear_result_test = (dot(theta[1:].T, test_data) + theta[0].reshape(k,1))
    logistic_result_test = (dot(weight[1:].T, test_data) + weight[0].reshape(k,1))
    
    y_test = y_test.T
    linear_result_test = linear_result_test.T
    logistic_result_test = logistic_result_test.T
    
    correct_count_linear = 0;
    corect_count_logistic = 0;
    for i in range(m):
        if (argmax(y_test[i]) == argmax(linear_result_test[i])):
            correct_count_linear = correct_count_linear + 1;
        if (argmax(y_test[i]) == argmax(logistic_result_test[i])):
            corect_count_logistic = corect_count_logistic + 1;
            
    print("------- Performance report on training data -------")
    print("Linear regression on test set", float(correct_count_linear)/float(len(y)) * 100)
    print("Logistic to regression on test set", float(corect_count_logistic)/float(len(y)) * 100)
    
# part5()
    
    




