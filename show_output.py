#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command line arguments:
directory_name


Created on Fri Jun  2 14:58:47 2017

@author: drake
"""

from train import build_net, get_inputs, get_masks, format_nsmask, get_nsmasks, BLACK
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
from scipy import misc
import argparse

TIME_STAMP = 20160414162830
PRINT_ALL = False
SHOW = True

BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])
GRAY = np.array([192, 192, 192])

# BLUE_FOR_GRAY (for example) means our net gave blue when the target mask
# gave gray
BLUE_FOR_GRAY = [85, 0, 0]  # Very dark red
BLUE_FOR_WHITE = [170, 0, 0]  # Dark red
GRAY_FOR_WHITE = [255, 0, 0]  # Bright red
GRAY_FOR_BLUE = [0, 85, 0]  # Dark green
WHITE_FOR_BLUE = [0, 170, 0]  # Medium green
WHITE_FOR_GRAY = [0, 255, 0]  # Bright green


def show_comparison_image(output, target, path):
    """Shows image of where output differs from target, color-coded by how they
    agree or disagree. Destructively modifies target."""
    target[np.logical_and((output == BLUE).all(axis=2), (target == GRAY).all(axis=2))] = BLUE_FOR_GRAY
    target[np.logical_and((output == BLUE).all(axis=2), (target == WHITE).all(axis=2))] = BLUE_FOR_WHITE
    target[np.logical_and((output == GRAY).all(axis=2), (target == BLUE).all(axis=2))] = GRAY_FOR_BLUE
    target[np.logical_and((output == GRAY).all(axis=2), (target == WHITE).all(axis=2))] = GRAY_FOR_WHITE
    target[np.logical_and((output == WHITE).all(axis=2), (target == BLUE).all(axis=2))] = WHITE_FOR_BLUE
    target[np.logical_and((output == WHITE).all(axis=2), (target == GRAY).all(axis=2))] = WHITE_FOR_GRAY
    disp = Image.fromarray(target.astype('uint8'))
    show_and_save(disp, path)


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

def show_and_save(img, path):
    if SHOW:
        img.show()
    img.save(path)
    
def load_net(train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy, result_dir, num_iterations):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num_iterations))
        inputs = get_inputs([TIME_STAMP])
        ns_vals = get_nsmasks([TIME_STAMP])
        img = out_to_image(y.eval(feed_dict={x: inputs, ns:ns_vals}))[0]
        if(PRINT_ALL):
            mask = np.array(misc.imread('data/simplemask/simplemask' + str(TIME_STAMP) + '.png'))
            show_and_save(Image.fromarray(inputs[0].astype('uint8')), result_dir + 'input.png')
            show_and_save(Image.fromarray(mask.astype('uint8')), result_dir + 'correct.png')
            show_comparison_image(img, mask, result_dir + 'comparison.png')
        img = Image.fromarray(img.astype('uint8'))
        show_and_save(img, result_dir + 'net-output.png')
        accuracy = accuracy.eval(feed_dict={x: inputs, y_: get_masks([TIME_STAMP]), ns: ns_vals})
        print('Accuracy = ' + str(accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--compare', help='If typed, images used to compare output will be displayed', action='store_true')
    parser.add_argument('--noshow', help='If typed, images are merely saved, not displayed', action='store_true')
    parser.add_argument('--time', help='Sets the time stamp to be loaded', type=int)
    args = parser.parse_args()
    print 
    if args.time:
        TIME_STAMP = args.time
    if args.compare:
        PRINT_ALL = True
    if args.noshow:
        SHOW = False
    dir_name = "results/" + args.directory + "/"        
    args = read_parameters(dir_name)
    step_version = read_last_iteration_number(dir_name)
    kernel_width = int(args['Kernel width'])
    layer_sizes = list(map(int, args['Layer sizes'].split()))
    load_net(*build_net(0, kernel_width, layer_sizes), dir_name, step_version)

