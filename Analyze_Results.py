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


#function to compare 1 image to annother and return the clor the image agrees on and red where they dissagree
#these compare the mask to the end image
def make_compared_image(result, mask):
    mask[(result != mask).any(axis=2)] = [255, 0 , 0]
    mask = Image.fromarray(mask.astype('uint8'))
    mask.show()
    return mask

def get_valid_stamps():
    with open('data/valid.stamps', 'rb') as f:
        valid_stamps = pickle.load(f)
    valid_stamps = valid_stamps[:10]
    return valid_stamps

def get_masks(time_stamps):
    masks = np.empty((len(time_stamps), 480, 480, 3))
    for i, s in enumerate(time_stamps):
        masks[i] = np.array(misc.imread('data/simplemask/simplemask' + str(s) + '.png'))
    return masks
        
#function that returns number of pixels that dissagree between two images
def find_num_disagreeing_pixels(result, mask):
    return np.sum((result == mask).all(axis=2))

# function that finds the worst results (maybe like 4), takes a set of time stamps
def find_worst_results(time_stamps = None, directory = "job_number_2_learning_rate_0.0001_layer_sizes_32_32_20170615162911", step_version = 3):
    time_stamps = get_valid_stamps()
    results = load_stamps(*build_net(), 'results/' + str(directory) + '/', step_version, time_stamps)
    
    
    print (results)
    for r in results:
        
        mask = Image.fromarray(r.astype('uint8'))
        mask.show()
        
    #masks = get_masks(time_stamps)
    #for i in results:
        

if __name__ == '__main__':
    #before = os.getcwd()
    #os.chdir('data')
    img1 = np.array(misc.imread('data/simplemask/' + 'simplemask20160414162830.png'))
    img2 = np.array(misc.imread('results/test3/' + 'net-output.png'))
    make_compared_image(img1, img2)
#    
#    load_stamps(*build_net(), 'results/' + str(sys.argv[1]) + '/', sys.argv[2])
    find_worst_results()
    