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

BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])
GRAY = np.array([192, 192, 192])
GREEN = np.array([0, 255, 0])
# BLUE_FOR_GRAY (for example) means our net gave blue when the target mask
# gave gray
BLUE_FOR_GRAY = [85, 0, 0]  # Very dark red
BLUE_FOR_WHITE = [170, 0, 0]  # Dark red
GRAY_FOR_WHITE = [255, 0, 0]  # Bright red
GRAY_FOR_BLUE = [0, 85, 0]  # Dark green
WHITE_FOR_BLUE = [0, 170, 0]  # Medium green
WHITE_FOR_GRAY = [0, 255, 0]  # Bright green


def show_comparison_images(outputs, targets):
    """Shows images of where outputs differ targets, color-coded by how they
    agree or disagree. Destructively modifies targets."""
    targets[np.logical_and((outputs == BLUE).all(axis=2), (targets == GRAY).all(axis=2))] = BLUE_FOR_GRAY
    targets[np.logical_and((outputs == BLUE).all(axis=2), (targets == WHITE).all(axis=2))] = BLUE_FOR_WHITE
    targets[np.logical_and((outputs == GRAY).all(axis=2), (targets == BLUE).all(axis=2))] = GRAY_FOR_BLUE
    targets[np.logical_and((outputs == GRAY).all(axis=2), (targets == WHITE).all(axis=2))] = GRAY_FOR_WHITE
    targets[np.logical_and((outputs == WHITE).all(axis=2), (targets == BLUE).all(axis=2))] = WHITE_FOR_BLUE
    targets[np.logical_and((outputs == WHITE).all(axis=2), (targets == GRAY).all(axis=2))] = WHITE_FOR_GRAY
    disp = Image.fromarray(targets.astype('uint8'))
    disp.show()


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

def one_hot_to_mask_1(max_indices, output):
    """Modifies (and returns) img to have sensible colors in place of
    one-hot vectors."""
    out = np.zeros([len(output), 480, 480 ,3])
    out[(max_indices == 0)] = BLUE
    out[(max_indices == 1)] = BLACK
    return out

def one_hot_to_mask_2(max_indices, output):
    """Modifies (and returns) img to have sensible colors in place of
    one-hot vectors."""
    out = np.zeros([len(output), 480, 480 ,3])
    out[(max_indices == 0)] = WHITE
    out[(max_indices == 1)] = BLUE
    out[(max_indices == 2)] = GRAY
    out[(max_indices == 3)] = BLACK
    out[(max_indices == 4)] = GREEN
    return out

def out_to_image_1(output):
    """Modifies (and returns) the output of the network for the 0th image as a
    human-readable RGB image."""
    output = output.reshape([-1, 480, 480, 2])
    # We use argmax instead of softmax so that we really will get one-hots
    max_indexes = np.argmax(output, axis=3)
    return one_hot_to_mask_1(max_indexes, output)

def out_to_image_2(output):
    """Modifies (and returns) the output of the network for the 0th image as a
    human-readable RGB image."""
    output = output.reshape([-1, 480, 480, 4])
    # We use argmax instead of softmax so that we really will get one-hots
    max_indexes = np.argmax(output, axis=3)
    return one_hot_to_mask_2(max_indexes, output)

def out_to_image(out1, out2):
    out1 = out1.reshape([-1, 480, 480, 2])
    max1 = np.argmax(out1, axis=3)
    print (max1)
    print (np.shape(max1))
    format1 = np.zeros([len(max1),480,480,1])
    format1[max1 == 1.0] = [10000000.0]
    out2 = out2.reshape([-1, 480, 480, 3])
    out = np.concatenate((out2,format1), axis=3)
    max_indexes = np.argmax(out, axis=3)
    return one_hot_to_mask_2(max_indexes, out)

def load_net(train_step_1, train_step_2, accuracy_1, accuracy_2, saver, init, x, y1, y_1, cross_entropy_1, y2, y_2, cross_entropy_2, ns_masks, disp1, disp2, result_dir, num_iterations):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num_iterations))
        inputs = get_inputs([TIME_STAMP])
        ns_mask = get_nsmasks([TIME_STAMP])
        out1 = disp1.eval(feed_dict={x: inputs})
        out2 = disp2.eval(feed_dict={x: inputs, ns_masks: ns_mask})
        img1 = out_to_image_1(out1)[0]
        img2 = out_to_image_2(y2.eval(feed_dict={x: inputs, ns_masks: ns_mask}))[0]
        img3 = out_to_image(out1, out2)[0]
        
        if(PRINT_ALL):
            mask = np.array(misc.imread('data/simplemask/simplemask' + str(TIME_STAMP) + '.png'))
            Image.fromarray(inputs[0].astype('uint8')).show()
            Image.fromarray(mask.astype('uint8')).show()
            show_comparison_images(img, mask)
        img1 = Image.fromarray(img1.astype('uint8'))
        img2 = Image.fromarray(img2.astype('uint8'))
        print (np.shape(img3))
        img3 = Image.fromarray(img3.astype('uint8'))
        img1.show()
        img2.show()
        img3.show()
        img1.save(result_dir + 'net-output.png')
        
        accuracy_1 = accuracy_1.eval(feed_dict={x: inputs, y_1: get_masks([TIME_STAMP], 1)})
        print('Accuracy = ' + str(accuracy_1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--compare', help='If typed, images used to compare output will be displayed', action='store_true')
    parser.add_argument('--time', help='Sets the time stamp to be loaded', type=int)
    args = parser.parse_args()
    print 
    if args.time:
        TIME_STAMP = args.time
    
    if args.compare:
        PRINT_ALL = True
    
    dir_name = "results/" + args.directory + "/"        
    args = read_parameters(dir_name)
    step_version = read_last_iteration_number(dir_name)
    layer_info = args['Layer info'].split()
    
    load_net(*build_net(layer_info, 0), dir_name, step_version)

