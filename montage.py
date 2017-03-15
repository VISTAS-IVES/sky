# -*- coding: utf-8 -*-
"""
Created  Jan  2017
Modified by gorr from Kadenze utililities 
https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info
@author: gorr

Place this file in the same folder as your Tensorflow files.  
"""
import numpy as np
import matplotlib.pyplot as plt

# Method to display an array of images in a single large image
def montage_images(images):
    #get the width and height of an image in the list of images
    img_h = images.shape[1]
    img_w = images.shape[2]
    # we want to fit all images into a large square. Calc the number of
    #   images per side of the square. If there are 100 images, then n_plots
    # will be 10
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    # Create a 2d array, m, to hold the square.  Assume 1 pixel will separate
    # each image. If each image width is 28, then the width of m will be
    # 28*n_plots + (nplots + 1) where the (nplots+1) corresponds to the 
    # 1 pixel of separation. We assume the image is grayscale and so has 
    # one color channel. If this is a color image, one needs to adjust 
    # m to have 3 channels.  The initial value of each pixel is 1. The
    # images will overwrite this value, leaving the 1 pixel separation at 
    # 1
    m = np.zeros(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    print('min in m: ',np.min(m), 'max in m: ',np.max(m))
    return m
    
def montage_filters(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.

    Parameters
    ----------
    W : Tensor
        Input tensor to create montage of.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    minW = np.min(W)
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
#    m = np.ones(
#        (W.shape[0] * n_plots + n_plots + 1,
#         W.shape[1] * n_plots + n_plots + 1)) * np.mean(W)
    m = np.full(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1),minW, dtype=np.float32)
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    print('min in m: ',np.min(m), 'max in m: ',np.max(m))
    return m
    