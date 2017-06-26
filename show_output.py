#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command line arguments:
directory_name


Created on Fri Jun  2 14:58:47 2017

@author: drake
"""

from net import build_net, get_inputs, format_nsmask, get_nsmasks, WHITE, BLUE, BLACK, GRAY
import numpy as np
import sys
import tensorflow as tf
from PIL import Image

def read_in(directory):
    F = open(directory + 'parameters.txt',"r")
    file = F.readlines()
    args = []
    for line in file:
        args.append(line[line.index("\t")+len("\t"):line.index("\n")])
    return args

def get_recent_step_version(directory):
    F = open(directory + 'output.txt',"r")
    file = F.readlines()
    line = file[len(file)-1]
    return int((line.split()[0]))

def one_hot_to_mask(max_indices, output):
    """Modifies (and returns) img to have sensible colors in place of
    one-hot vectors."""
    out = np.zeros([len(output), 480, 480 ,3])
    out[(max_indices == 0)] = WHITE
    out[(max_indices == 1)] = BLUE
    out[(max_indices == 2)] = GRAY
    out[(max_indices == 3)] = BLACK
    return out

def out_to_image(output):
    """Modifies (and returns) the output of the network for the 0th image as a
    human-readable RGB image."""
    output = output.reshape([-1, 480, 480, 4])
    # We use argmax instead of softmax so that we really will get one-hots
    max_indexes = np.argmax(output, axis=3)
    return one_hot_to_mask(max_indexes, output)

def load_net(train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy, result_dir, num_iterations):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num_iterations))
        inputs = get_inputs([20160414162830])
        ns_vals = get_nsmasks([20160414162830])
        img = out_to_image(y.eval(feed_dict={x: inputs, ns:ns_vals}))[0]
        img = Image.fromarray(img.astype('uint8'))
        img.show()
        img.save(result_dir + 'net-output.png')

if __name__ == '__main__':
    dir_name = "results/" + sys.argv[1] + "/"
    args = read_in(dir_name)
    step_version = get_recent_step_version(dir_name)
    kernel_width = int(args[2])
    layer_sizes = list(map(int,args[3].split()))
    load_net(*build_net(0, kernel_width, layer_sizes), dir_name, step_version)
