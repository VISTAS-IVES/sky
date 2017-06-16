#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:32:13 2017

@author: jeffmullins
"""

from net import build_net, get_inputs
from load_net import out_to_image, load_stamps
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
        
def find_num_disagreeing_pixels(results, mask):
    return np.sum((results != mask).any(axis=2)) / (480*480)

def find_worst_results(num_worst, time_stamps, directory, step_version):
    time_stamps = get_valid_stamps()
    results = load_stamps(*build_net(), directory, step_version, time_stamps)
    masks = get_masks(time_stamps)
    num_inconsistent = np.zeros(len(time_stamps))
    for i in range(len(time_stamps)):
        num_inconsistent[i] = find_num_disagreeing_pixels(results[i], masks[i])
    indices = num_inconsistent.argsort()[num_worst*-1:][::-1]
    print (np.take(num_inconsistent, indices))
    return np.take(time_stamps, indices)

     
def display_sky_images(time_stamps):
    for s in time_stamps:
        Image.fromarray(np.array(misc.imread('data/simpleimage/simpleimage' + str(s) + '.jpg'))).show()
        

if __name__ == '__main__':
    time_stamps = get_valid_stamps()
    directory = 'results/' + sys.argv[1] + '/'
    step_version = int(sys.argv[2])
    
    worst_time_stamps = find_worst_results(5, time_stamps, directory, step_version)
    
    results = load_stamps(*build_net(), directory, step_version, worst_time_stamps)
    masks = get_masks(worst_time_stamps)
    make_compared_images(results, masks)
