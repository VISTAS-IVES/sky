#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
from PIL import Image

# Define colors
BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])
# Distances from center of an image
BATCH_SIZE = 50

def one_hot_to_mask(max_indexs, output):
    """Modifies (and returns) img to have sensible colors in place of
    one-hot vectors."""
    output[(max_indexs == 0)] = WHITE
    output[(max_indexs == 1)] = BLUE
    output[(max_indexs == 2)] = BLACK
    return output

def out_to_image(output, n):
    """Modifies (and returns) the output of the network for the nth image as a
    human-readable RGB image."""
    output = output.reshape([-1,480,480,3])[n]
    outs = output
    # We use argmax instead of softmax so that we really will get one-hots
    max_indexes = np.argmax(outs, axis = 2)
    return one_hot_to_mask(max_indexes, outs)


def scale(img):
    for r in range(480):
        for c in range(480):
            img[r, c] = img[r, c] / 255.0
    return img


def make_b_mask_boolean(img):
    black_mask = np.zeros((480,480), dtype=bool)
    black_mask[(img == BLACK).all(axis=2)] = True
    return black_mask


def give_b_mask_black_values(bool_mask):
    black_mask = np.zeros((480,480,3), dtype=np.float32)
    black_mask[bool_mask] = [0.0, 0.0, 10000000.0]
    return black_mask

def mask_to_one_hot(img):
    """Modifies (and returns) img to have a one-hot vector for each
    pixel."""
    img[(img == WHITE).all(axis=2)] = np.array([1, 0, 0])
    img[(img == BLUE).all(axis=2)]  = np.array([0, 1, 0])
    img[(img == BLACK).all(axis=2)] = np.array([0, 0, 1])
    return img


def mask_to_index(img):
    """Returns a new version of img with an index (expected color)
    for each pixel."""
    result = np.ndarray(shape=[img.shape[0], img.shape[1]])
    result[(img == WHITE).all(axis=2)] = 0
    result[(img == BLUE).all(axis=2)]  = 1
    result[(img == BLACK).all(axis=2)] = 2
    return result


def get_inputs(stamps):
    inputs = np.empty((len(stamps), 480, 480, 3))
    for i, s in enumerate(stamps):
        img = np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))
        #img = scale(img)
        inputs[i] = img
        # inputs[i] = np.concatenate((img, RADII), axis=2)
    return inputs


def get_masks(stamps):
    masks = np.empty((len(stamps), 480, 480))
    for i, s in enumerate(stamps):
        masks[i] = mask_to_index(np.array(misc.imread('data/simplemask/simplemask' + str(s) + '.png')))
    return masks.reshape((-1))


def weight_variable(shape, n_inputs):
    initial = tf.truncated_normal(shape, stddev = 2/math.sqrt(n_inputs))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    #initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def load_validation_batch():
    with open('data/valid.stamps', 'rb') as f:
        valid_stamps = pickle.load(f)
    valid_stamps = valid_stamps[:50]
    valid_inputs = get_inputs(valid_stamps)
    valid_correct = get_masks(valid_stamps)
    return valid_inputs, valid_correct


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
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def convo_layer(num_in, num_out, prev, relu = True):
    W = weight_variable([3, 3, num_in, num_out], 3 * 3 * num_in)
    b = bias_variable([num_out])
    if relu: 
        h = tf.nn.relu(conv2d(prev, W) + b)
    else:
        h = conv2d(prev, W) + b
    return h


def mask_layer(last_layer, b_mask):
    btf_mask = tf.constant(b_mask)
    return tf.add(btf_mask, last_layer)


def build_net(learning_rate = 0.0001, layer_sizes = [32, 32]):
    print("Building network")
    bool_mask = make_b_mask_boolean(misc.imread('data/always_black_mask.png'))
    b_mask = give_b_mask_black_values(bool_mask)
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    num_layers = len(layer_sizes)+1
    h = [None] * (num_layers)
    h[0] = convo_layer(3, layer_sizes[0], x)
    for i in range(1, num_layers-1):
        h[i] = convo_layer(layer_sizes[i-1], layer_sizes[i], h[i-1])
    h[num_layers-1] = convo_layer(layer_sizes[num_layers-2], 3, h[num_layers-2], False)
    m = mask_layer(h[num_layers-1], b_mask)
    y = tf.reshape(m, [-1, 3])
    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    return train_step, accuracy, saver, init, x, y, y_, cross_entropy


def train_net(train_step, accuracy, saver, init, x, y, y_, cross_entropy,
              valid_inputs, valid_correct, result_dir):
    print("training network")
    start = time.time()
    # Get image and make the mask into a one-hotted mask
#    with open('data/train.stamps', 'rb') as f:
#        train_stamps = pickle.load(f)
    with open(result_dir + 'output.txt', 'w') as f:
        with tf.Session() as sess:
            init.run()
            print('Step\tTrain\tValid', file=f, flush=True)
            batch = (20160414162830,)
            inputs = get_inputs(batch)
            correct = get_masks(batch)
            for i in range(1, 100000 + 1):
                # batch = random.sample(train_stamps, BATCH_SIZE)
                train_step.run(feed_dict={x: inputs, y_: correct})
                if i % 100 == 0:
                    saver.save(sess, result_dir + 'weights', global_step=i)
                    train_accuracy = accuracy.eval(feed_dict={
                            x: inputs, y_: correct})
                    #entropy = cross_entropy.eval(feed_dict={
                            #x: inputs, y_: correct})
                    # valid_accuracy = accuracy.eval(feed_dict={
                    # x:valid_inputs, y_:valid_correct})
                    print('{}\t{:1.5f}'.format(i, train_accuracy), file=f, flush=True)
                    #print('{}\t{:1.5f}'.format(i, train_accuracy))

                    # print('{}\t{:1.5f}\t{:1.5f}'.format(i, train_accuracy, valid_accuracy), file=f, flush=True)
                    
        stop = time.time()
        print('Elapsed time: {} seconds'.format(stop - start), file=f, flush=True)


if __name__ == '__main__':
    job_number = sys.argv[1]
    learning_rate = float(sys.argv[2])
    layer_sizes = sys.argv[3::]
    layer_sizes_print = '_'.join(layer_sizes)
    out_dir = 'results/job_number_' + job_number + '_' + 'learning_rate_' + str(learning_rate) + '_' + 'layer_sizes_' + layer_sizes_print + '_' + datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    os.makedirs(out_dir)
    layer_sizes = list(map(int, layer_sizes))
    train_net(*build_net(learning_rate, layer_sizes), *load_validation_batch(), out_dir)
        
