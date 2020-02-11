import matplotlib as test
test.use('Agg')

from numpy import *
import os
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


import tensorflow as tf

from auto_pick import *

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])




train_x = zeros((1, 227, 227,3)).astype(float32)
xdim = train_x.shape[1:] 
net_data = load("bvlc_alexnet.npy").item()
x = tf.placeholder(tf.float32, (None,) + xdim)

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                depth_radius=radius,
                                                alpha=alpha,
                                                beta=beta,
                                                bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                depth_radius=radius,
                                                alpha=alpha,
                                                beta=beta,
                                                bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)



# Classification layer.
x2 = tf.placeholder(tf.float32, [None, 64896])
    
# Number of hidden units.
nhid = 300
    
W0 = tf.Variable(tf.random_normal([64896, nhid], stddev=0.00001))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.00001))
    
W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.00001))
b1 = tf.Variable(tf.random_normal([6], stddev=0.00001))

layer1 = tf.nn.tanh(tf.matmul(x2, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1
    
y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])

lam = 0
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL) # training.


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# Gets the correct answer.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Input data
#train_dir = "part10_data/train_data/";
#train_data = os.listdir(train_dir)
#name = train_data[0]
#print(train_dir + name)
#img = (imread(train_dir + name)[:,:,:3]).astype(float32)
#img = img - mean(img)
#img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
#img = (imread("laska.png")[:,:,:3]).astype(float32)
#img = img - mean(img)
#img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]

# test = sess.run(conv4, feed_dict = {x:[img]}) 
#print("img.shape", img.shape)
#print("conv1.shape", (sess.run(conv1, feed_dict = {x:[img]})).shape)
#print("conv2.shape", (sess.run(conv2, feed_dict = {x:[img]})).shape)
#print("conv3.shape", (sess.run(conv3, feed_dict = {x:[img]})).shape)
#print("conv4.shape", (sess.run(conv4, feed_dict = {x:[img]})).shape)
# print("conv4W.shape", (sess.run(conv4W, feed_dict = {x:[img]})).shape)
# print("conv4b.shape", (sess.run(conv4b, feed_dict = {x:[img]})).shape)
# print("conv3W.shape", (sess.run(conv3W, feed_dict = {x:[img]})).shape)
# print("conv3b.shape", (sess.run(conv3b, feed_dict = {x:[img]})).shape)
# print("conv2W.shape", (sess.run(conv2W, feed_dict = {x:[img]})).shape)
# print("conv2b.shape", (sess.run(conv2b, feed_dict = {x:[img]})).shape)
# print("conv1W.shape", (sess.run(conv1W, feed_dict = {x:[img]})).shape)
# print("conv1b.shape", (sess.run(conv1b, feed_dict = {x:[img]})).shape)



def pick_data():
    train_dir = "part10_data/train_data/"
    valid_dir = "part10_data/valid_data/"
    test_dir = "part10_data/test_data/"

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
    pickrandom(act, "cropped_p10/", train_dir, test_dir, valid_dir, 60, 30, 30)


#Get the train data.
def get_train_data():
    xs = array([])
    ys = array([])
    train_dir = "part10_data/train_data/";


    train_data = os.listdir(train_dir)

    for name in train_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            
            img = (imread(train_dir + name)[:,:,:3]).astype(float32)
            img = img - mean(img)
            img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
            
            if xs.size == 0:
                xs = vstack([(sess.run(conv4, feed_dict = {x:[img]})).flatten()])

            elif xs.size > 0:
                xs = vstack([xs, (sess.run(conv4, feed_dict = {x:[img]})).flatten()])

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
    test_dir = "part10_data/test_data/";


    test_data = os.listdir(test_dir)

    for name in test_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            
            img = (imread(test_dir + name)[:,:,:3]).astype(float32)
            img = img - mean(img)
            img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
            
            if xs.size == 0:
                xs = vstack([(sess.run(conv4, feed_dict = {x:[img]})).flatten()])
            elif xs.size > 0:
                xs = vstack([xs, (sess.run(conv4, feed_dict = {x:[img]})).flatten()])


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




# Get the valid data
def get_valid_data():
    xs = array([])
    ys = array([])
    valid_dir = "part10_data/valid_data/";


    valid_data = os.listdir(valid_dir)

    for name in valid_data:
        if any([actor.split()[1].lower() in name for actor in act]):
            
            img = (imread(valid_dir + name)[:,:,:3]).astype(float32)
            img = img - mean(img)
            img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
            
            if xs.size == 0:
                xs = vstack([(sess.run(conv4, feed_dict = {x:[img]})).flatten()])
            elif xs.size > 0:
                xs = vstack([xs, (sess.run(conv4, feed_dict = {x:[img]})).flatten()])


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


# W0 has size 64896 * nhid = 64896 * 30
# def part_11(W0):
#     W = W0.T;
#     for i in range(W.shape[0]):
#         file_name = "img10_" + str(i)
#         img = W[i].reshape(13, 13, 384)
#         imsave("part11_output/" + file_name, img)
        
def part_11():
    pick_data()
    
    # get train data extracted from CONV4 layer.
    xs, ys = get_train_data()
    
    # gets the validation data extracted from CONV4 layer.
    valid_x, valid_y =  get_valid_data()
    
    # gets the training data extracted from CONV4 layer.
    test_x, test_y = get_test_data()
    
    # arrays for plotting graph.
    y_test = [] 
    y_train = []
    y_valid = []
    
    for i in range(10):
        sess.run(train_step, feed_dict={x2: xs, y_: ys})
        test_set_performance = sess.run(accuracy, feed_dict={x2: test_x, y_: test_y})
        train_set_performance = sess.run(accuracy, feed_dict={x2: xs, y_: ys})
        valid_set_performance = sess.run(accuracy, feed_dict={x2: valid_x, y_: valid_y})
       
        print("i=", i)
        print("Performance on the test set: ", test_set_performance)
        print("Performance on the train set: ", train_set_performance)
        print("Performance on the validation set:",  valid_set_performance)
        
        y_test.append(test_set_performance * 100)
        y_train.append(train_set_performance * 100)
        y_valid.append(valid_set_performance * 100)
    print((sess.run(layer1, feed_dict={x2: xs, y_: ys})).shape)
    

part_11()








    
