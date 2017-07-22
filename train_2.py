#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line arguments:
job_number layer_size_1 ... layer_size_n

Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson

"""
from train import build_net
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
GREEN = np.array([0, 255, 0])

# Distances from center of an image
BATCH_SIZE = 5
LEARNING_RATE = 0.0001

def check_for_commit():
    label = subprocess.check_output(["git", "status", "--untracked-files=no", "--porcelain"])
    if (0 != (str(label).count('\\n'))):
        raise Exception('Not in clean git state\n')


def read_parameters(directory):
    """Reads the parameters.txt file in directory. Returns a dictionary
    associating labels with keys."""
    F = open(directory + 'parameters.txt', 'r')
    file = F.readlines()
    args = {}
    for line in file:
        key, value = line.split(':\t')
        args[key] = value
    return args


def read_last_iteration_number(directory):
    """Reads the output.txt file in directory. Returns the iteration number
    on the last row."""
    F = open(directory + 'output.txt', 'r')
    file = F.readlines()
    line = file[len(file) - 1]
    return (line.split()[0])

def out_to_black(output):
    """Modifies (and returns) the output of the network for the 0th image as a
    human-readable RGB image."""
    output = output.reshape([BATCH_SIZE, 480, 480, 2])
    # We use argmax instead of softmax so that we really will get one-hots
    max_indexes = np.argmax(output, axis=3)
    ret = np.zeros([BATCH_SIZE, 480, 480, 1])
    ret[(max_indexes == 1).all(axis=2)] = [10000000.0]
    return ret

def load_net(train_step, accuracy, saver, init, x, y, y_, cross_entropy, result_dir, num_iterations):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num_iterations))
        return {"train_step": train_step, "accuracy": accuracy, "saver": saver, "init": init, "x": x, "y": y, "y_": y_, "cross_entropy": cross_entropy, "result_dir": result_dir, "num_iterations": num_iterations}

def save_params(job_number, learning_rate, layer_info, out_dir):
    F = open(out_dir + 'parameters.txt',"w+")
    F.write("Job number:\t" + str(job_number) + "\n")
    F.write("Learning rate:\t" + str(learning_rate) + "\n")
    F.write("Layer info:\t" + ' '.join(layer_info) + "\n")
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
    result[(img == GREEN).all(axis=2)] = 3
    result[(img == BLACK).all(axis=2)] = 4

    return result


def get_inputs(stamps):
    """Returns a tensor of images specified by stamps. Dimensions are: image,
    row, column, color."""
    inputs = np.empty((len(stamps), 480, 480, 3))
    for i, s in enumerate(stamps):
        img = np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))
        inputs[i] = img
    return inputs


#def mask_layer(last_layer, b_mask, g_mask):
#    gtf_mask = tf.constant(g_mask)
#    return tf.add(gtf_mask, tf.add(b_mask, last_layer))

def make_b_mask_boolean(img):
    """Takes a black/non-black image and return a corresponding True/False mask."""
    black_mask = np.zeros((480, 480), dtype=bool)
    black_mask[(img != BLUE).any(axis=2)] = True
    return black_mask


def give_b_mask_black_values(bool_mask):
    """Takes a boolean mask and returns a 3-channel image with [0, 0, 1e7]
    where the mask is True, [0, 0, 0] elsewhere."""
    black_mask = np.zeros((480, 480, 1), dtype=np.float32)
    black_mask[bool_mask] = [10000000.0]
    return black_mask


def give_g_mask_green_values(bool_mask):
    """Takes a boolean mask and returns a 3-channel image with [0, 0, 1e7]
    where the mask is True, [0, 0, 0] elsewhere."""
    black_mask = np.zeros((480, 480, 4), dtype=np.float32)
    black_mask[bool_mask] = [0.0, 0.0, 0.0, 10000000.0]
    return black_mask


def get_masks(stamps):
    """Returns a tensor of correct label categories specified by stamps.
    Dimensons are image, row, column. The tensor has been flattened into a
    single vector."""
    masks = np.empty((len(stamps), 480, 480))
    for i, s in enumerate(stamps):
        masks[i] = mask_to_index(np.array(misc.imread('data/greenmask/greenmask' + str(s) + '.png')))
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
        raise ValueError('number of features({}) is not ' +
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def max_pool_layer(prev, pool_width, pool_height, name):
    pool_width, pool_height = int(pool_width), int(pool_height)
    return tf.nn.max_pool(prev,[1, pool_width, pool_height, 1],strides=[1, 1, 1, 1], padding='SAME', name = name)

def convo_layer(num_in, num_out, width, prev, name, relu=True):
    num_in, num_out, width = int(num_in), int(num_out), int(width)
    with tf.variable_scope(name):
        initial = tf.truncated_normal([width, width, num_in, num_out], stddev=(2 / math.sqrt(width * width * num_in)))
        W = tf.get_variable("weights_1", initializer = initial)
        initial = tf.constant(0.1, shape=[num_out])
        b = tf.Variable(initial, name='biases')
        if relu:
            h = tf.nn.relu(conv2d(prev, W, name) + b)
        else:
            h = conv2d(prev, W, name) + b   
    return h

def mask_layer(last_layer, btf_mask, gtf_mask):
    added = tf.add(gtf_mask, last_layer)
    return tf.concat([added, btf_mask], 3)

def parse_layer_info(layer_info):
    table = {}
    count = 0
    name = ""
    for layer in layer_info:
        hold1 = layer.split(":")
        name = hold1[0]
        hold2 = hold1[1].split("-")
        oper = hold2[0]
        args = hold2[1:]
        info = {}
        
        if oper == 'conv' :
            kernel = args[0]
            outs = args[1]
            tf_name = name
            if count == 0:
                ins = 3
                tf_name = "hidden0_1"
            else:
                ins = table[args[2]]['outs']
            prev_name = args[2]    
            info = {"outs":outs, "ins":ins, "kernel":kernel, "prev":prev_name, "tf_name":tf_name}
            
        if oper == 'maxpool' :
            pool_width = args[0]
            pool_height = args[1]
            outs = (table[args[2]])["outs"]
            prev_name = args[2]
            info = {"outs":outs, "pool_width":pool_width, "pool_height":pool_height, "prev":prev_name}
        
        if oper == 'concat' :
            outs = str(int(table[args[0]]["outs"]) + int(table[args[1]]["outs"]))
            prev_1 = args[0]
            prev_2 = args[1]
            info = {"outs":outs, "prev_1":prev_1, "prev_2":prev_2}
            
        table[name] = info
        count += 1 
    return table, name

def get_name_oper(layer):
    hold = layer.split(":")
    name = hold[0]
    oper = hold[1].split("-")[0]
    return name, oper
    


def build_net_two(layer_info, net_one_layer_info, dir_name1, step_version, learning_rate=0.0001):
    print("Building network")
    tf.reset_default_graph()
    
    net1_args = load_net(*build_net(net_one_layer_info, 0), dir_name1, step_version)
    
    bool_mask = make_b_mask_boolean(misc.imread('data/b_mask.png'))    
    b_mask = tf.placeholder(tf.float32, [None, 480, 480, 1])
    
    bool_mask = make_b_mask_boolean(misc.imread('data/g_mask.png'))
    g_mask = give_g_mask_green_values(bool_mask)
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    num_layers = len(layer_info)
    table, last_name = parse_layer_info(layer_info)
    h = {}
    h["in"] = x
    for n in range(0, num_layers-1):
        # make first layer
        name, oper = get_name_oper(layer_info[n])
        if (oper == 'conv'):
            h[name] = convo_layer(table[name]["ins"], table[name]["outs"], table[name]["kernel"], h[table[name]["prev"]], table[name]["tf_name"])
            
        if (oper == 'maxpool'):
            h[name] = max_pool_layer(h[table[name]["prev"]], table[name]["pool_width"], table[name]["pool_height"], name)
            
        if (oper == 'concat'):
                h[name] = tf.concat([h[table[name]["prev_1"]], h[table[name]["prev_2"]]], 3)
 
    h[last_name] = convo_layer(table[last_name]["ins"], table[last_name]["outs"], table[last_name]["kernel"], h[table[last_name]["prev"]], table[last_name]["tf_name"], False)
    m = mask_layer(h[last_name], b_mask, g_mask)
    y = tf.reshape(m, [-1, 5])
    y_ = tf.placeholder(tf.int64, [None])
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    saver = tf.train.Saver()
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()    
    return train_step, accuracy, saver, init, x, y, y_, cross_entropy, b_mask, net1_args


def train_net(train_step, accuracy, saver, init, x, y, y_, cross_entropy, b_mask, net1_args,
              valid_inputs, valid_correct, result_dir,
              dir_name1, num_iterations, net_one_layer_info):
        
        print("Training network")
        start = time.time()
        # Get image and make the mask into a one-hotted mask
        with open('data/train.stamps', 'rb') as f:
            train_stamps = pickle.load(f)
        with open(result_dir + 'output.txt', 'w') as f:
            with tf.Session() as sess:
                init.run()
                print('Step\tTrain\tValid', file=f, flush=True)
                j = 0
                for i in range(1, 10 + 1):
                    j += 1
                    if (j*BATCH_SIZE >= len(train_stamps)):
                        j = 1
                    #batch = random.sample(train_stamps, BATCH_SIZE)
                    batch = train_stamps[(j-1)*BATCH_SIZE : j*BATCH_SIZE]
                    inputs = get_inputs(batch)
                    correct = get_masks(batch)
                    ns_masks = net1_args["y"].eval(feed_dict={net1_args["x"]: inputs})
                    ns_masks = out_to_black(ns_masks)
                    train_step.run(feed_dict={x: inputs, y_: correct, b_mask: ns_masks})
                    if i % 1 == 0:
                        saver.save(sess, result_dir + 'weights', global_step=i)
    #                    train_accuracy = accuracy.eval(feed_dict={
    #                            x: inputs, y_: correct, ns: ns_vals})
    #                    valid_accuracy = accuracy.eval(feed_dict={
    #                            x: valid_inputs, y_: valid_correct, ns: valid_ns_vals})
                        train_accuracy = accuracy.eval(feed_dict={
                                x: inputs, y_: correct, b_mask: ns_masks})
                        valid_accuracy = accuracy.eval(feed_dict={
                                x: valid_inputs, y_: valid_correct, b_mask: ns_masks})
                        print (":-)")
                        print('{}\t{:1.5f}\t{:1.5f}'.format(i, train_accuracy, valid_accuracy), file=f, flush=True)                             
    #                    print('{}\t{:1.5f}\t{:1.5f}'.format(i, train_accuracy, valid_accuracy))  
            stop = time.time()  
            F = open(out_dir + 'parameters.txt',"a")
            F.write("Elapsed time:\t" + str(stop - start) + " seconds\n")
            F.close()




if __name__ == '__main__':
 #   check_for_commit()
    tim = time.time()
    print(tim)
    job_number = str(tim)
#    job_number = sys.argv[1]
    layer_info = sys.argv[3::]
    layer_sizes_print = '_'.join(layer_info)
    out_dir = 'results/exp' + job_number + '/'
    os.makedirs(out_dir)
    save_params(job_number, LEARNING_RATE, layer_info, out_dir)
    #layer_info = list(map(int, layer_info))
    
    dir_name1 = 'results/' + sys.argv[2] + '/'
    args = read_parameters(dir_name1)
    step_version = read_last_iteration_number(dir_name1)
    net_one_layer_info = args['Layer info'].split()
    train_net(*build_net_two(layer_info, net_one_layer_info, dir_name1, step_version, LEARNING_RATE), *load_validation_batch(), out_dir, dir_name1, step_version, net_one_layer_info)