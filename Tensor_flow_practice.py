#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson

"""
import numpy as np
from scipy import misc
from PIL import Image
import tensorflow as tf
import os
import math
import time
import random
import pickle

# Define colors
BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])  

def mask_to_one_hot(img):
    """Modifies (and returns) img to have a one-hot vector for each
    pixel."""
    img[(img == WHITE).all(axis = 2)] = np.array([1,0,0])
    img[(img == BLUE).all(axis = 2)]  = np.array([0,1,0])
    img[(img == BLACK).all(axis = 2)] = np.array([0,0,1]) 
    return img

def mask_to_index(img):
    """Returns a new version of img with an index (expected color)
    for each pixel."""
    result = np.ndarray(shape = [img.shape[0], img.shape[1]])
    result[(img == WHITE).all(axis = 2)] = 0
    result[(img == BLUE).all(axis = 2)]  = 1
    result[(img == BLACK).all(axis = 2)] = 2
    return result

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

def get_inputs(stamps):
    inputs = np.empty((len(stamps), 480, 480, 3))
    for i, s in enumerate(stamps):  
        inputs[i] = np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))
    return inputs

def get_masks(stamps):
    masks = np.empty((len(stamps), 480, 480))
    for i, s in enumerate(stamps):  
        masks[i] = mask_to_index(np.array(misc.imread('data/simplemask/simplemask' + str(s) + '.png')))
    return masks.reshape((-1))

def weight_variable(shape, n_inputs):
    initial = tf.truncated_normal(shape, stddev=2 / math.sqrt(n_inputs))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
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

def convo_layer(num_in, num_out, prev):
    W = weight_variable([3, 3, num_in, num_out], 3 * 3 * num_in)
    b = bias_variable([num_out])
    h = tf.nn.relu(conv2d(prev, W) + b)
    return h

def build_net():
    print ("Building network")
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    h1 = convo_layer(3,32,x)
    h2 = convo_layer(32,32,h1)
    h3 = convo_layer(32,3,h2)
    y = tf.reshape(h3, [-1, 3])
    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    return train_step, accuracy, saver, init, x, y, y_

def train_net(train_step, accuracy, saver, init, x, y, y_, valid_inputs, valid_correct):
    start = time.time()
    # Get image and make the mask into a one-hotted mask
    with open('data/train.stamps', 'rb') as f:
        train_stamps = pickle.load(f)
    with tf.Session() as sess:
        init.run()
        print('Step\tTrain\tValid')
        for i in range(1, 1000 + 1):
            batch = random.sample(train_stamps, 50)
            inputs = get_inputs(batch)
            correct = get_masks(batch)
            train_step.run(feed_dict={x: inputs, y_: correct})
            if i % 10 == 0:
                saver.save(sess, 'results/weights', global_step=i)
                train_accuracy = accuracy.eval(feed_dict={
                        x:inputs, y_:correct})
                valid_accuracy = accuracy.eval(feed_dict={
                        x:valid_inputs, y_:valid_correct})
                print('{}\t{:1.5f}\t{:1.5f}'.format(i, train_accuracy, valid_accuracy))
    stop = time.time()
    print('Elapsed time: {} seconds'.format(stop - start))
   
def test_net(train_step, accuracy, saver, init, x, y, y_, valid_inputs, valid_correct, num):
     # Train
    with tf.Session() as sess:
        saver.restore(sess, 'results/weights-' + str(num))
        valid_accuracy = accuracy.eval(feed_dict={
                x:valid_inputs, y_:valid_correct})
        print('{:1.5f}'.format(valid_accuracy))
        inputs = get_inputs([20160414162830])
        img = out_to_image(y.eval(feed_dict={x: inputs}), 0)
        img = Image.fromarray(img.astype('uint8'))
        img.show()
#        img.save('data/out_masks/output-' + str(i).zfill(6) + '.png')
             
if __name__ == '__main__':
    train_net(*build_net(), *load_validation_batch())
#   test_net(*build_net(), *load_validation_batch(), 770)