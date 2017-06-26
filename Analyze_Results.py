#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line arguments:
directory_name


Created on Thu Jun 15 15:32:13 2017

@author: jeffmullins
"""

from net import build_net, get_inputs, get_nsmasks
from load_net import out_to_image
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
from scipy import misc
import pickle
from net import BATCH_SIZE

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


def show_comparison_images(outputs, targets):
    """Shows images of where outputs differ targets, color-coded by how they
    agree or disagree. Destructively modifies targets."""
    for i in range(len(outputs)):
        targets[i][np.logical_and((outputs[i] == BLUE).all(axis=2), (targets[i] == GRAY).all(axis=2))] = BLUE_FOR_GRAY
        targets[i][np.logical_and((outputs[i] == BLUE).all(axis=2), (targets[i] == WHITE).all(axis=2))] = BLUE_FOR_WHITE
        targets[i][np.logical_and((outputs[i] == GRAY).all(axis=2), (targets[i] == BLUE).all(axis=2))] = GRAY_FOR_BLUE
        targets[i][np.logical_and((outputs[i] == GRAY).all(axis=2), (targets[i] == WHITE).all(axis=2))] = GRAY_FOR_WHITE
        targets[i][np.logical_and((outputs[i] == WHITE).all(axis=2), (targets[i] == BLUE).all(axis=2))] = WHITE_FOR_BLUE
        targets[i][np.logical_and((outputs[i] == WHITE).all(axis=2), (targets[i] == GRAY).all(axis=2))] = WHITE_FOR_GRAY
        disp = Image.fromarray(targets[i].astype('uint8'))
        disp.show()


def read_valid_stamps():
    """Reads the valid.stamps file in data and returns a list of stamps."""
    with open('data/valid.stamps', 'rb') as f:
        valid_stamps = pickle.load(f)
    valid_stamps = valid_stamps[:BATCH_SIZE]
    return valid_stamps


def read_target(time_stamp):
    """Reads and returns the target mask corresponding to time_stamps from
    the simplemask directory."""
    return np.array(misc.imread('data/simplemask/simplemask' + str(time_stamp) + '.png'))


def read_targets(time_stamps):
    """Reads and returns the target masks corresponding to time_stamps from
    the simplemask directory."""
    masks = np.empty((len(time_stamps), 480, 480, 3))
    for i, s in enumerate(time_stamps):
        masks[i] = read_target(s)
    return masks


def disagreement_rate(output, target):
    """Returns the proportion of pixels in output that disagree with target."""
    return np.sum((output != target).any(axis=2)) / (480*480)


def load_stamps(train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy, result_dir, num, stamps):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num))
        inputs = get_inputs(stamps)
        ns_vals = get_nsmasks(stamps)
        outputs = out_to_image(y.eval(feed_dict={x: inputs, ns: ns_vals}))
        return outputs.reshape(-1, 480, 480, 3)

def find_worst_results(num_worst, time_stamps, directory, step_version, kernel, layers):
    train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy = build_net(kernel_width = kernel, layer_sizes = layers)
    with tf.Session() as sess:
        saver.restore(sess, directory + 'weights-' + str(step_version))
        time_stamps = read_valid_stamps()
        num_inconsistent = np.zeros(len(time_stamps))
        for i, s in enumerate(time_stamps):
            inputs = get_inputs([s])
            ns_vals = get_nsmasks([s])
            result = out_to_image(y.eval(feed_dict={x: inputs, ns: ns_vals}))
            result = result.reshape(480, 480, 3)
            mask = read_target(s)
            num_inconsistent[i] = disagreement_rate(result, mask)
        indices = num_inconsistent.argsort()[num_worst*-1:][::-1]
        print ("Worst results percentages:\t" + str(np.take(num_inconsistent, indices)))
    return np.take(time_stamps, indices)

     
def display_sky_images(time_stamps):
    for s in time_stamps:
        Image.fromarray(np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))).show()


if __name__ == '__main__':
    time_stamps = read_valid_stamps()
    dir_name = "results/" + sys.argv[1] + "/"
    args = read_parameters(dir_name)
    step_version = read_last_iteration_number(dir_name)
    kernel_width = int(args['Kernel width'])
    layer_sizes = list(map(int, args['Layer sizes'].split()))
    worst_time_stamps = find_worst_results(5, time_stamps, dir_name, step_version, kernel_width, layer_sizes)
    print ("Worst time stamps:\t" + str(worst_time_stamps))
    outputs = load_stamps(*build_net(0, kernel_width, layer_sizes), dir_name, step_version, worst_time_stamps)
    targets = read_targets(worst_time_stamps)
    show_comparison_images(outputs, targets)
