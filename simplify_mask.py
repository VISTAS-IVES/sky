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
import shutil
from pathlib import Path

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

def make_random_sample(size, in1 = "data/simplified_images/20160414/", out1 = "data/simplified_images/test10/",
                       in2 = "data/simplified_masks/20160414/", out2 = "data/simplified_masks/test10/"):
    """Copies a random set of size files from in1 to out1 and a corresponding set from
    in2 to out2."""
    files = np.array(os.listdir(in1))
    rand_indices = np.random.choice(range(len(files)), size, replace=False)
#    rand_indices = np.random.randint(0,high = len(files),size = size)
    files = np.take(files,rand_indices)
    for f in files:
        shutil.copy(in1 + f, out1 + f)
    files = np.take(np.array(os.listdir(in2)), rand_indices)
    for f in files:
        shutil.copy(in2 + f, out2 + f)

def find_failed_correspondences(images='data/images/20160415/', masks='data/masks/20160415/'):
    """Prints names of images files that do not have matching mask files."""
    for f in os.listdir(images):
        date = f[40:48]
        time = f[48:54]
        g = Path(masks + "sgptsicldmaskC1.a1." + date + "." + time + ".png." + date + time + ".png")
        if not g.is_file():
            print (f)
            Path(images + f).unlink()
    
#if __name__ == '__main__':
#    print (simplify_images('data/images/20160415/', 'data/simplified_images/20160415/'))
#    print (simplify_masks('data/masks/20160415/', 'data/simplified_masks/20160415/'))
