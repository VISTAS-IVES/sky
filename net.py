#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line arguments:
job_number learning_rate kernel_width layer_size_1 ... layer_size_n

Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson

"""

import numpy as np
from scipy import misc
from datetime import datetime
import tensorflow as tf
import sys
import os
import math
import time
import random
import pickle
import subprocess
import tensorflow.contrib.slim as slim
from math import sqrt
from PIL import Image

# Define colors
BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])
GRAY = np.array([192, 192, 192])

# Distances from center of an image
BATCH_SIZE = 50

def check_for_commit():
    label = subprocess.check_output(["git", "status", "--untracked-files=no", "--porcelain"])
    if (0 != (str(label).count('\\n'))):
        raise Exception('Not in clean git state\n')

def save_params(job_number, learning_rate, kernel_width, layer_sizes, out_dir):
    F = open(out_dir + 'parameters.txt',"w+")
    F.write("Job number:\t" + str(job_number) + "\n")
    F.write("Learning rate:\t" + str(learning_rate) + "\n")
    F.write("Kernel width:\t" + str(kernel_width) + "\n")
    F.write("Layer sizes:\t" + ' '.join(layer_sizes) + "\n")
    label = subprocess.check_output(["git", "rev-parse", "HEAD"])
    F.write("Git commit:\t" + str(label)[2:-3:] + "\n")
    F.close()

def scale(img):
    for r in range(480):
        for c in range(480):
            img[r, c] = img[r, c] / 255.0
    return img


def mask_to_one_hot(img):
    """Modifies (and returns) img to have a one-hot vector for each
    pixel."""
    img[(img == WHITE).all(axis=2)] = np.array([1, 0, 0, 0])
    img[(img == BLUE).all(axis=2)] =  np.array([0, 1, 0, 0])
    img[(img == GRAY).all(axis=2)] =  np.array([0, 0, 1, 0])
    img[(img == BLACK).all(axis=2)] = np.array([0, 0, 0, 1])
    
    return img


def mask_to_index(img):
    """Returns a new version of img with an index (expected color)
    for each pixel."""
    result = np.ndarray(shape=[img.shape[0], img.shape[1]])
    result[(img == WHITE).all(axis=2)] = 0
    result[(img == BLUE).all(axis=2)] = 1
    result[(img == GRAY).all(axis=2)] = 2
    result[(img == BLACK).all(axis=2)] = 3
    return result


def get_inputs(stamps):
    """Returns a tensor of images specified by stamps. Dimensions are: image,
    row, column, color."""
    inputs = np.empty((len(stamps), 480, 480, 3))
    for i, s in enumerate(stamps):
        img = np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))
        inputs[i] = img
    return inputs


def get_masks(stamps):
    """Returns a tensor of correct label categories specified by stamps.
    Dimensons are image, row, column. The tensor has been flattened into a
    single vector."""
    masks = np.empty((len(stamps), 480, 480))
    for i, s in enumerate(stamps):
        masks[i] = mask_to_index(np.array(misc.imread('data/simplemask/simplemask' + str(s) + '.png')))
    return masks.reshape((-1))

def format_nsmask(img):
    """Takes a boolean mask and returns a 1-channel image with [0, 0, 1e7]
    where the mask is True, [0, 0, 0] elsewhere."""
    ns_mask = np.full((480, 480, 1), -1000000.0, dtype=np.float32)
    ns_mask[(img == BLACK).all(axis=2)] = [10000000.0]
    return ns_mask

def get_nsmasks(stamps):
    masks = np.empty((len(stamps), 480, 480, 1))
    for i, s in enumerate(stamps):
        masks[i] = format_nsmask((np.array(misc.imread('data/nsmask/nsmask' + str(s) + '.png'))))
    return masks

def weight_variable(shape, n_inputs, num):
    initial = tf.truncated_normal(shape, stddev=(2 / math.sqrt(n_inputs)))
    with tf.name_scope('conv' + str(num)):
        W = tf.Variable(initial, name = 'weights')
    return W


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, num):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv'+str(num))


def load_validation_batch():
    with open('data/valid.stamps', 'rb') as f:
        valid_stamps = pickle.load(f)
    valid_stamps = valid_stamps[:BATCH_SIZE]
    valid_inputs = get_inputs(valid_stamps)
    valid_correct = get_masks(valid_stamps)
    valid_ns_vals = get_nsmasks(valid_stamps)
    return valid_inputs, valid_correct, valid_ns_vals


def max_out(inputs, num_units, axis=None):
    # This function was taken from the following link
    # https://github.com/philipperemy/tensorflow-maxout
    # For information about licensing see the LICENSE file
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not ' +
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

#old convo layer code
#def convo_layer(num_in, num_out, width, prev, num, relu=True):
#    W = weight_variable([width, width, num_in, num_out], width * width * num_in, num)
#    b = bias_variable([num_out])
#    if relu:
#        h = tf.nn.relu(conv2d(prev, W, num) + b)
#    else:
#        h = conv2d(prev, W, num) + b
#    return h


def convo_layer(num_in, num_out, width, prev, num, relu=True):
    
    with tf.variable_scope('hidden' + str(num)):
        initial = tf.truncated_normal([width, width, num_in, num_out], stddev=(2 / math.sqrt(width * width * num_in)))
        W = tf.get_variable("weights", initializer = initial)
        initial = tf.constant(0.1, shape=[num_out])
        b = tf.Variable(initial, name='biases')
        if relu:
            h = tf.nn.relu(conv2d(prev, W, num) + b)
        else:
            h = conv2d(prev, W, num) + b   
    return h

      

def mask_layer(last_layer, ns_vals):
    #ns_vals = tf.constant(ns_vals)
    return tf.concat([last_layer, ns_vals], 3)

def build_net(learning_rate=0.0001, kernel_width = 3, layer_sizes=[32, 32]):
    print("Building network")
    tf.reset_default_graph()
    ns = tf.placeholder(dtype = tf.float32, shape = (None, 480, 480 ,1))
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    num_layers = len(layer_sizes)+1
    h = [None] * (num_layers)
    if (num_layers > 1):
        h[0] = convo_layer(3, layer_sizes[0], kernel_width, x, 0)
        for i in range(1, num_layers-1):
            h[i] = convo_layer(layer_sizes[i-1], layer_sizes[i], kernel_width, h[i-1], i)
        h[num_layers-1] = convo_layer(layer_sizes[num_layers-2], 3, kernel_width, h[num_layers-2], num_layers-1, False)
    else:
        h[0] = convo_layer(3, 3, kernel_width, x, 0, False)
    m = mask_layer(h[num_layers-1], ns)
    y = tf.reshape(m, [-1, 4])
    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    saver = tf.train.Saver()
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    
    return train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy


def train_net(train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy,
              valid_inputs, valid_correct, valid_ns_vals, result_dir):
    print("Training network")
    start = time.time()
    # Get image and make the mask into a one-hotted mask
    with open('data/train.stamps', 'rb') as f:
        train_stamps = pickle.load(f)
    with open(result_dir + 'output.txt', 'w') as f:
        with tf.Session() as sess:
            init.run()
            print('Step\tTrain\tValid', file=f, flush=True)
            for i in range(1, 1000 + 1):
                batch = random.sample(train_stamps, BATCH_SIZE)
                inputs = get_inputs(batch)
                correct = get_masks(batch)
                ns_vals = get_nsmasks(batch)
                train_step.run(feed_dict={x: inputs, y_: correct, ns: ns_vals})
                if i % 50 == 0:
                    saver.save(sess, result_dir + 'weights', global_step=i)
                    train_accuracy = accuracy.eval(feed_dict={
                            x: inputs, y_: correct, ns: ns_vals})
                    valid_accuracy = accuracy.eval(feed_dict={
                            x: valid_inputs, y_: valid_correct, ns: valid_ns_vals})

                    print('{}\t{:1.5f}\t{:1.5f}'.format(i, train_accuracy, valid_accuracy), file=f, flush=True)                             
            
        stop = time.time()  
        F = open(out_dir + 'parameters.txt',"a")
        F.write("Elapsed time:\t" + str(stop - start) + " seconds\n")
        F.close()

if __name__ == '__main__':
    check_for_commit()
    job_number = sys.argv[1]
    learning_rate = float(sys.argv[2])
    kernel_width = int(sys.argv[3])
    layer_sizes = sys.argv[4::]
    layer_sizes_print = '_'.join(layer_sizes)
    out_dir = 'results/exp' + job_number + '/'
    os.makedirs(out_dir)
    save_params(job_number, learning_rate, kernel_width, layer_sizes, out_dir)
    layer_sizes = list(map(int, layer_sizes))
    train_net(*build_net(learning_rate, kernel_width, layer_sizes), *load_validation_batch(), out_dir)