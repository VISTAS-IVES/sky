#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:23:23 2017

@author: drake
"""

import numpy as np
from scipy import misc

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

img = misc.imread('stripes.png')
print(list_colors(img))
