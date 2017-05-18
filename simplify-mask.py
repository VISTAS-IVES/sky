#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:23:23 2017

@author: drake
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc

#img = mpimg.imread('Shades_of_green.png')
#img = mpimg.imread('search-3-512.jpg')
img = misc.imread('Stripes.png')
imgplot = plt.imshow(img)
plt.show()

def find_colors():
    global img
    
    w,y,z = np.shape(img)
    
    a = img.reshape(y*w,z)
    print (np.shape(a))
    print (a)
    """
    new_img = np.array([tuple(row) for row in new_img])
    print (np.shape(new_img))
    colors = np.unique(new_img)
    #colors = set( tuple(v) for c in img for v in c )
    print (np.shape(new_img))
"""
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = a[idx]
 
    return unique_a
