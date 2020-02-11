import matplotlib as test
test.use('Agg')

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
import tensorflow as tf

act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'];


# ----------------------------------------------------------------
# PART 7
#-----------------------------------------------------------------
# helper function for converting to gray scale.
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

# Get the train data.
def get_train_data():
    xs = array([])
    ys = array([])
    train_dir = "part7_data/train_data/";


    train_data = os.listdir(train_dir)

    for name in train_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            img = imread(train_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            
            if xs.size == 0:
                xs = vstack([img])
            elif xs.size > 0:
                xs = vstack([xs, img])


            one_hot = zeros(6)
            # assign y based on label of the trainning images.
            if "drescher" in name:
                one_hot[0] = 1
            elif "ferrera" in name:
                one_hot[1] = 1
            elif "chenoweth" in name:
                one_hot[2] = 1
            elif "baldwin" in name:
                one_hot[3] = 1
            elif "hader" in name:
                one_hot[4] = 1
            elif "carell" in name:
                one_hot[5] = 1

            if ys.size == 0:
                ys = vstack([one_hot])
            elif ys.size != 0:
                ys = vstack([ys, one_hot])
    return xs, ys


# Get the test data
def get_test_data():
    xs = array([])
    ys = array([])
    test_dir = "part7_data/test_data/";


    test_data = os.listdir(test_dir)

    for name in test_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            img = imread(test_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            
            if xs.size == 0:
                xs = vstack([img])
            elif xs.size > 0:
                xs = vstack([xs, img])


            one_hot = zeros(6)
            # assign y based on label of the trainning images.
            if "drescher" in name:
                one_hot[0] = 1
            elif "ferrera" in name:
                one_hot[1] = 1
            elif "chenoweth" in name:
                one_hot[2] = 1
            elif "baldwin" in name:
                one_hot[3] = 1
            elif "hader" in name:
                one_hot[4] = 1
            elif "carell" in name:
                one_hot[5] = 1

            if ys.size == 0:
                ys = vstack([one_hot])
            elif ys.size != 0:
                ys = vstack([ys, one_hot])
    return xs, ys

# plot graph part7
def plot_graph_part7(y_test, y_train, y_valid): 
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    
    plt.subplot(2,1,1)
    plt.plot(y_test, '-', color='red', lw = 2, label="test set")
    plt.plot(y_train, ':', color='green', lw = 2, label=" train set")
    plt.plot(y_valid, '--', color='blue', lw = 2, label=" validation set")
    plt.ylabel('Correctness rate (%)')
    plt.xlabel('Number of iterations')
    plt.title('Learning curves.')
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
    plt.savefig("plot_part7.png")
    

    
    
# Get the valid data
def get_valid_data():
    xs = array([])
    ys = array([])
    valid_dir = "part7_data/valid_data/";


    valid_data = os.listdir(valid_dir)

    for name in valid_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            img = imread(valid_dir + name)
            img = rgb2gray(img)
            img = img.flatten()
            
            if xs.size == 0:
                xs = vstack([img])
            elif xs.size > 0:
                xs = vstack([xs, img])


            one_hot = zeros(6)
            # assign y based on label of the trainning images.
            if "drescher" in name:
                one_hot[0] = 1
            elif "ferrera" in name:
                one_hot[1] = 1
            elif "chenoweth" in name:
                one_hot[2] = 1
            elif "baldwin" in name:
                one_hot[3] = 1
            elif "hader" in name:
                one_hot[4] = 1
            elif "carell" in name:
                one_hot[5] = 1

            if ys.size == 0:
                ys = vstack([one_hot])
            elif ys.size != 0:
                ys = vstack([ys, one_hot])
    return xs, ys



def pick_data():
    train_dir = "part7_data/train_data/"
    valid_dir = "part7_data/valid_data/"
    test_dir = "part7_data/test_data/"

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(valid_dir): 
        shutil.rmtree(valid_dir)
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    os.makedirs(test_dir)

    # Randomly pick 60 images for trainning, 30 images for testing and validating.
    pickrandom(act, "cropped/", train_dir, test_dir, valid_dir, 60, 30, 30)



def part7():
    pick_data()
    
    # 1024 pixels for each of the image.
    x = tf.placeholder(tf.float32, [None, 1024])
    
    # Number of hidden units.
    nhid = 30
    
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])
    
    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL) # training.
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    # Gets the correct answer.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    # gets the training data.
    test_x, test_y = get_test_data()
    
    # gets the validation data.
    valid_x, valid_y =  get_valid_data()
    
    # get train data.
    xs, ys = get_train_data()
    
    # arrays for plotting graph.
    y_test = [] 
    y_train = []
    y_valid = []
    
    for i in range(500):
    # Trains the model.
        sess.run(train_step, feed_dict={x: xs, y_: ys})
        test_set_performance = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        train_set_performance = sess.run(accuracy, feed_dict={x: xs, y_: ys})
        valid_set_performance = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
        
        print("i=", i)
        print("Performance on the test set: ", test_set_performance)
        print("Performance on the train set: ", train_set_performance)
        print("Performance on the validation set:",  valid_set_performance)
    
    
    
        y_test.append(test_set_performance * 100)
        y_train.append(train_set_performance * 100)
        y_valid.append(valid_set_performance * 100)
    
    
    plot_graph_part7(y_test, y_train, y_valid)


# part7()



# ----------------------------------------------------------------
# PART 8
#-----------------------------------------------------------------
def pick_data_part8():
    train_dir = "part8_data/train_data/"
    valid_dir = "part8_data/valid_data/"
    test_dir = "part8_data/test_data/"

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(valid_dir): 
        shutil.rmtree(valid_dir)
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir)
    os.makedirs(valid_dir)
    os.makedirs(test_dir)

    # Randomly pick 10 images for trainning, 30 images for testing and validating.
    pickrandom(act, "cropped/", train_dir, test_dir, valid_dir, 8, 40, 40)


def plot_graph_part8(y_test, y_train, y_valid, file_name): 
    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(y_test, '-', color='red', lw = 2, label="test set")
    plt.plot(y_train, ':', color='green', lw = 2, label=" train set")
    plt.plot(y_valid, '--', color='blue', lw = 2, label=" validation set")
    plt.ylabel('Correctness rate (%)')
    plt.xlabel('Number of iterations')
    plt.title('Learning curves.')
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
    plt.savefig(file_name)


def part8_with_no_regularization():
    pick_data_part8()
    
    # 1024 pixels for each of the image.
    x = tf.placeholder(tf.float32, [None, 1024])
    
    # Number of hidden units.
    nhid = 2000
    
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])
    
    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL) # training.
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    # Gets the correct answer.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    # gets the training data.
    test_x, test_y = get_test_data()
    
    # gets the validation data.
    valid_x, valid_y =  get_valid_data()
    
    # get train data.
    xs, ys = get_train_data()
    
    # arrays for plotting graph.
    y_test = [] 
    y_train = []
    y_valid = []
    
    for i in range(100):
    # Trains the model.
        sess.run(train_step, feed_dict={x: xs, y_: ys})
        test_set_performance = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        train_set_performance = sess.run(accuracy, feed_dict={x: xs, y_: ys})
        valid_set_performance = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
        
        # print("i=", i)
        
        y_test.append(test_set_performance * 100)
        y_train.append(train_set_performance * 100)
        y_valid.append(valid_set_performance * 100)
    
    plot_graph_part8(y_test, y_train, y_valid, "plot_part8_noreg.png")
    
    print("-----With no regularization------")
    print("Performance on the test set: ", test_set_performance)
    print("Performance on the train set: ", train_set_performance)
    print("Performance on the validation set:",  valid_set_performance)
    print("--------")
# part8_with_no_regularization()



def part8_with_regularization():
    pick_data_part8()
    
    # 1024 pixels for each of the image.
    x = tf.placeholder(tf.float32, [None, 1024])
    
    # Number of hidden units.
    nhid = 2000
    
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])
    
    lam = 0.8
    decay_penalty = lam*tf.reduce_sum(tf.square(W0)) + lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y)) + decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL) # training.
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    # Gets the correct answer.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    # gets the training data.
    test_x, test_y = get_test_data()
    
    # gets the validation data.
    valid_x, valid_y =  get_valid_data()
    
    # get train data.
    xs, ys = get_train_data()
    
    # arrays for plotting graph.
    y_test = [] 
    y_train = []
    y_valid = []
    
    for i in range(100):
    # Trains the model.
        sess.run(train_step, feed_dict={x: xs, y_: ys})
        test_set_performance = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        train_set_performance = sess.run(accuracy, feed_dict={x: xs, y_: ys})
        valid_set_performance = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
        
        # print("i=", i)
        y_test.append(test_set_performance * 100)
        y_train.append(train_set_performance * 100)
        y_valid.append(valid_set_performance * 100)
    
    print("-----With-regularization------")
    print("Performance on the test set: ", test_set_performance)
    print("Performance on the train set: ", train_set_performance)
    print("Performance on the validation set:",  valid_set_performance)
    print("-------------")
    plot_graph_part8(y_test, y_train, y_valid, "plot_part8_reg.png")
    
# part8_with_regularization()




# ----------------------------------------------------------------
# PART 9
#-----------------------------------------------------------------
def part9():
    pick_data_part8()
    
    # 1024 pixels for each of the image.
    x = tf.placeholder(tf.float32, [None, 1024])
    
    # Number of hidden units.
    nhid = 40
    
    W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
    
    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])
    
    lam = 10
    decay_penalty = lam*tf.reduce_sum(tf.square(W0)) + lam*tf.reduce_sum(tf.square(W1))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y)) + decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL) # training.
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    # Gets the correct answer.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # get train data.
    xs, ys = get_train_data()
    
    for i in range(50):
        # Trains the model.
        sess.run(train_step, feed_dict={x: xs, y_: ys})

    # 1024 * nhid = 1024 * 40
    W0 = sess.run(W0, feed_dict={x: xs, y_: ys})
    
    # 40 * 6
    W1 = sess.run(W1, feed_dict={x: xs, y_: ys})
    
    # 6 * 40
    W1 = W1.T
    
    # Step 1: Find 6 hiddens unit that are most sensitive.
    unit = zeros(6)
    unit[0] = argmax(W1[0])
    unit[1] = argmax(W1[1])
    unit[2] = argmax(W1[2])
    unit[3] = argmax(W1[3])
    unit[4] = argmax(W1[4])
    unit[5] = argmax(W1[5])
    
    # 40 * 1024
    W0 = W0.T
    
    # Step 2: Extract 6 weights come into the 6 sensitive neurons.
    img0 = imresize((W0[unit[0]]).reshape(32, 32), (64, 64))
    img1 = imresize((W0[unit[1]]).reshape(32, 32), (64, 64))
    img2 = imresize((W0[unit[2]]).reshape(32, 32), (64, 64))
    img3 = imresize((W0[unit[3]]).reshape(32, 32), (64, 64))
    img4 = imresize((W0[unit[4]]).reshape(32, 32), (64, 64))
    img5 = imresize((W0[unit[5]]).reshape(32, 32), (64, 64))
    
    imsave("part9_img0.png", img0)
    imsave("part9_img1.png", img1)
    imsave("part9_img2.png", img2)    
    imsave("part9_img3.png", img3)
    imsave("part9_img4.png", img4)
    imsave("part9_img5.png", img5)

#part9()