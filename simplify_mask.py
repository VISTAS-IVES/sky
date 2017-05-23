#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:23:23 2017

@author: drake
"""
import numpy as np
from scipy import misc
import os
from PIL import Image


def list_colors(img):
    """Returns an array of the unique colors in img."""
    # Adapted from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    # Flatten img
    a = img.reshape(-1, img.shape[2])
    # Make this an array of colors (each of which is a length 4 array)
    b = np.ascontiguousarray(a) \
                    .view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    # Find the locations of the unique colors
    _, idx = np.unique(b, return_index = True)
    # Return those colors
    return a[idx]

# Colors from mask images
BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
GREEN = np.array([0, 255, 0])
GRAY = np.array([192, 192, 192])
YELLOW = np.array([255, 255, 0])
WHITE = np.array([255, 255, 255])

def simplify_colors(img):
    """Returns and image with GREEN and YELLOW pixels made black, GRAY pixels
    WHITE. Destructively modifies img."""
    img[(img == GREEN).all(axis = 2)] = BLACK
    img[(img == YELLOW).all(axis = 2)] = BLACK
    img[(img == GRAY).all(axis = 2)] = WHITE
    return img
    
def crop_image(img):
    #crops the mask dwon to 480 x 480
    return np.delete(img, np.concatenate((np.arange(80),np.arange(80)+560)), axis=0)

def simplify_masks(in_dir, out_dir):
    """Writes similified versions of all images in in_dir to out_dir.
    Returns an array of relative frequencies of BLUE, WHITE, and BLACK."""
    counts = np.zeros(3, dtype=np.int)
    for file in os.listdir(in_dir):
        img = misc.imread(in_dir + file)
        img = crop_image(img)
        simplified = simplify_colors(img)
        counts = counts + color_counts(simplified)
        Image.fromarray(simplified).save(out_dir + file)
    return counts / counts.sum()

def simplify_images(in_dir, out_dir):
    """Crops the images in in_dir down to 480x480 and writes those to out_dir
    and resturns the number of images cropped"""
    counts = 0
    for file in os.listdir(in_dir):
        if file[0] != ".":
            img = misc.imread(in_dir + file)
            cropped = crop_image(img)
            counts = counts + 1
            Image.fromarray(cropped).save(out_dir + file)
    return counts


def color_counts(img):
    """Returns an array of the number of BLUE, WHITE, and BLACK pixels
    in img."""
    blue = (img == BLUE).all(axis = 2).sum()
    white = (img == WHITE).all(axis = 2).sum()
    black = (img == BLACK).all(axis = 2).sum()
    return np.array([blue, white, black])
    
if __name__ == '__main__':
    print (simplify_images('data/images/', 'data/simplified_images/'))
    print (simplify_masks('data/masks/', 'data/simplified_masks/'))
