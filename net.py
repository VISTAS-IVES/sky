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

# Define colors
BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])  
# Distances from center of an image
RADII = np.empty((480,480, 1))
BATCH_SIZE = 50

"""inputs is a 480x480x3 array, we add the distance from the center
making it a 480x480x4 array""" 
for r in range(480):
    for c in range(480):
        RADII[r, c, 0] = (math.sqrt((239.5 - r) ** 2 + (239.5 - c) ** 2)) / (math.sqrt(2) * 239.5)

def scale(img):
    for r in range(480):
        for c in range(480):
            img[r,c] = img[r,c] / 255.0
    return img           

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

def get_inputs(stamps):
    inputs = np.empty((len(stamps), 480, 480, 3))
    for i, s in enumerate(stamps):
        img = np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))
        img = scale(img)
        #inputs[i] = np.concatenate((img, RADII), axis=2)
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

def first_convo_layer(num_in, num_out, prev, radii):
    W = weight_variable([3, 3, num_in, num_out], 3 * 3 * num_in)
    R = weight_variable([1, 1, 1, num_out], 1)
    b = bias_variable([num_out])
    h = tf.nn.relu(conv2d(prev, W) + b + conv2d(radii, R))
    return h

def convo_layer(num_in, num_out, prev):
    W = weight_variable([3, 3, num_in, num_out], 3 * 3 * num_in)
    b = bias_variable([num_out])
    h = tf.nn.relu(conv2d(prev, W) + b)
    return h

def build_net(learning_rate=1e-4):
    print ("Building network")
    tf.reset_default_graph()
    radii = tf.constant(np.reshape(RADII, (1, 480, 480, 1)),  dtype = tf.float32)
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    h1 = first_convo_layer(3,64,x, radii)
    h2 = convo_layer(64,32,h1)
    h3 = convo_layer(32,3,h2)
    y = tf.reshape(h3, [-1, 3])
    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    return train_step, accuracy, saver, init, x, y, y_

def train_net(train_step, accuracy, saver, init, x, y, y_, valid_inputs, valid_correct, result_dir):
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
            for i in range(1, 10000 + 1):
#                batch = random.sample(train_stamps, BATCH_SIZE)
                train_step.run(feed_dict={x: inputs, y_: correct})
                if i % 1000 == 0:
                    saver.save(sess, result_dir + 'weights', global_step=i)
                    train_accuracy = accuracy.eval(feed_dict={
                            x:inputs, y_:correct})
#                    valid_accuracy = accuracy.eval(feed_dict={
#                            x:valid_inputs, y_:valid_correct})
                    print('{}\t{:1.5f}'.format(i, train_accuracy), file=f, flush=True)
#                    print('{}\t{:1.5f}\t{:1.5f}'.format(i, train_accuracy, valid_accuracy), file=f, flush=True)
        stop = time.time()
        print('Elapsed time: {} seconds'.format(stop - start), file=f, flush=True)
             
if __name__ == '__main__':
    job_number = sys.argv[1]
    learning_rate = float(sys.argv[2])
    out_dir = 'results/job_number_' + job_number + '_' + 'learning_rate_' + str(learning_rate) + '_' +  datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    os.makedirs(out_dir)
    train_net(*build_net(learning_rate), *load_validation_batch(), out_dir)
