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
    """Makes a mask of the image to find GREEN values, then removes them"""
    mask = (img == GREEN).all(axis = 2)
    img[mask == True] = BLACK
    """Does the same for YELLOW values"""
    mask = (img == YELLOW).all(axis = 2)
    img[mask == True] = BLACK
    """Does the same for YELLOW values"""
    mask = (img == GRAY).all(axis = 2)
    img[mask == True] = WHITE
    return img
    
def simplify_images(in_dir, out_dir):
    """Writes similified versions of all images in in_dir to out_dir.
    Creates out_dir if necessary."""
    os.mkdir(out_dir)
    for file in os.listdir(in_dir):
        img = misc.imread(in_dir + file)
        new_img = simplify_colors(img)
        im = Image.fromarray(new_img)
        im.save(out_dir + file)
    
if __name__ == '__main__':
    pass
