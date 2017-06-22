#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command line arguments:
directory_name


Created on Thu Jun 15 15:32:13 2017

@author: jeffmullins
"""

from net import build_net, get_inputs
from load_net import out_to_image
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
import os
import pickle



#def make_compared_image(result, mask):
#    mask[(result != mask).any(axis=2)] = [255, 0 , 0]
#    mask = Image.fromarray(mask.astype('uint8'))
#    mask.show()
#    return mask


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
    return (line.split()[0])

def make_compared_images(results, masks):
    for i in range(len(results)):
        masks[i][(results[i] != masks[i]).any(axis=2)] = [255, 0 , 0]
        disp = Image.fromarray(masks[i].astype('uint8'))
        disp.show()
    return masks

def get_valid_stamps():
    with open('data/valid.stamps', 'rb') as f:
        valid_stamps = pickle.load(f)
    valid_stamps = valid_stamps[:50]
    return valid_stamps

def get_masks(time_stamps):
    masks = np.empty((len(time_stamps), 480, 480, 3))
    for i, s in enumerate(time_stamps):
        masks[i] = np.array(misc.imread('data/simplemask/simplemask' + str(s) + '.png'))
    return masks
     
def get_mask(time_stamp):
    return np.array(misc.imread('data/simplemask/simplemask' + str(time_stamp) + '.png'))

   
def find_num_disagreeing_pixels(results, mask):
    return np.sum((results != mask).any(axis=2)) / (480*480)


def load_stamps(train_step, accuracy, saver, init, x, y, y_, cross_entropy, result_dir, num, stamps):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num))
        inputs = get_inputs(stamps)
        outputs = out_to_image(y.eval(feed_dict={x: inputs}))
        return outputs.reshape(-1, 480, 480, 3)

def find_worst_results(num_worst, time_stamps, directory, step_version, kernel, layers):
    train_step, accuracy, saver, init, x, y, y_, cross_entropy = build_net(kernel_width = kernel, layer_sizes = layers)
    with tf.Session() as sess:
        saver.restore(sess, directory + 'weights-' + str(step_version))
        time_stamps = get_valid_stamps()
        num_inconsistent = np.zeros(len(time_stamps))
        for i, s in enumerate(time_stamps):
            inputs = get_inputs([s])
            result = out_to_image(y.eval(feed_dict={x: inputs}))
            result = result.reshape(480, 480, 3)
            mask = get_mask(s)
            num_inconsistent[i] = find_num_disagreeing_pixels(result, mask)
        indices = num_inconsistent.argsort()[num_worst*-1:][::-1]
        print ("Worst results percentages:\t" + str(np.take(num_inconsistent, indices)))
    return np.take(time_stamps, indices)

     
def display_sky_images(time_stamps):
    for s in time_stamps:
        Image.fromarray(np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))).show()


if __name__ == '__main__':
    
    time_stamps = get_valid_stamps()
    dir_name = "results/" + sys.argv[1] + "/"
    args = read_in(dir_name)
    step_version = get_recent_step_version(dir_name)
    kernel_width = int(args[2])
    layer_sizes = list(map(int,args[3].split()))
    
    worst_time_stamps = find_worst_results(5, time_stamps, dir_name, step_version, kernel_width, layer_sizes)
    print ("Worst time stamps:\t" + str(worst_time_stamps))
    results = load_stamps(*build_net(0, kernel_width, layer_sizes), dir_name, step_version, worst_time_stamps)
    masks = get_masks(worst_time_stamps)
    make_compared_images(results, masks)
