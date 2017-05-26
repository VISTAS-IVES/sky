#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess Total Sky Imager data from arm.gov. To use this:
    
1) Get the skyimage and cldmask data as described in Jessica's email.
2) Put the tars in a new directory called 'data' (which should live in
the same directory as this file -- git will ignore it).
3) Run this file.
    
This will create subfolders skyimage, cldmask, simpleimage, and simplemask.
Our program runs on the last two. It also creates three files test.stamps,
valid.stamps, and train.stamps; these are pickled lists of timestamp
numbers, which our program uses to load the appropriate files during
training.

NOTE: For simplicity, the functions in this file assume that you are running
them from the data directory. Running this file temporarily switches to that
directory.

Created on Fri May 26 10:45:02 2017

@author: drake
"""

import os
import tarfile
from scipy import misc
import numpy as np
from PIL import Image
import random
import pickle

BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
GREEN = np.array([0, 255, 0])
GRAY = np.array([192, 192, 192])
YELLOW = np.array([255, 255, 0])
WHITE = np.array([255, 255, 255])

def create_dirs():
    os.mkdir('skyimage')
    os.mkdir('cldmask')
    os.mkdir('simpleimage')
    os.mkdir('simplemask')

def unpack_tar(file, dir):
    """Given a tarfile file, moves it to dir, unpacks it, and deletes it."""
    g = dir + file
    os.rename(file, g)
    tar = tarfile.open(g)
    tar.extractall(path=dir)
    tar.close()
    # Sometimes this complains that g is still in use, so we're going to
    # keep trying until it works
    while True:
        try:
            os.remove(g)
            break
        except OSError as err:
            continue

def unpack_all_tars():
    for f in os.listdir('./'):
        if f.endswith('.tar'):
            if 'skyimage' in f:
                unpack_tar(f, 'skyimage/')
            elif 'cldmask' in f:
                unpack_tar(f, 'cldmask/')

def simplify_name(filename):
    """Simplifies the filenames we get from arm.gov."""
    return filename[6:filename.index('C1')] + filename[-18:]

def simplify_all_names():
    for dir in ('skyimage/', 'cldmask/'):
        for f in os.listdir(dir):
            os.rename(dir + f, dir + simplify_name(f))

def extract_timestamp(filename):
    """Assume filename ends in something like 20160415235930.jpg or
    20160415235930.png."""
    return filename[-18:-4]
    
def remove_images_without_matching_masks():
    """Deletes image files that do not have matching mask files."""
    for f in os.listdir('skyimage/'):
        g = 'cldmask/cldmask' + extract_timestamp(f) + '.png'
        if not os.path.isfile(g):
            os.remove('skyimage/' + f)
            
def crop_image(img):
    """Crops img down to 480 x 480."""
    return np.delete(img, np.concatenate((np.arange(80), np.arange(80)+560)), axis=0)

def simplify_images():
    """Crops the images in in_dir down to 480x480 and writes those to out_dir
    and returns the number of images cropped"""
    counts = 0
    for file in os.listdir('skyimage/'):
        if file[0] != '.':
            img = misc.imread('skyimage/' + file)
            cropped = crop_image(img)
            counts = counts + 1
            Image.fromarray(cropped).save('simpleimage/simpleimage' + extract_timestamp(file) + '.jpg')
    return counts

def color_counts(img):
    """Returns an array of the number of BLUE, WHITE, and BLACK pixels
    in img."""
    blue = (img == BLUE).all(axis = 2).sum()
    white = (img == WHITE).all(axis = 2).sum()
    black = (img == BLACK).all(axis = 2).sum()
    return np.array([blue, white, black])

def simplify_colors(img):
    """Returns an image with GREEN and YELLOW pixels made black, GRAY pixels
    WHITE. Destructively modifies img."""
    img[(img == GREEN).all(axis = 2)] = BLACK
    img[(img == YELLOW).all(axis = 2)] = BLACK
    img[(img == GRAY).all(axis = 2)] = WHITE
    return img

def simplify_masks():
    """Writes similified versions of all images in in_dir to out_dir.
    Returns an array of relative frequencies of BLUE, WHITE, and BLACK."""
    counts = np.zeros(3, dtype=np.int)
    for file in os.listdir('cldmask/'):
        img = misc.imread('cldmask/' + file)
        img = crop_image(img)
        simplified = simplify_colors(img)
        counts = counts + color_counts(simplified)
        Image.fromarray(simplified).save('simplemask/simplemask' + extract_timestamp(file) + '.png')
    return counts / counts.sum()

def separate_stamps(stamps):
    random.shuffle(stamps)
    test = stamps[0:int(len(stamps)*0.2)]
    valid = stamps[int(len(stamps)*0.2):int(len(stamps)*0.36)]
    train = stamps[int(len(stamps)*0.36):]
    return test, valid, train

def separate_data():
    stamps = [int(extract_timestamp(f)) for f in os.listdir('simpleimage/')]
    test, valid, train = separate_stamps(stamps)
    with open('test.stamps', 'wb') as f:
        pickle.dump(test, f)
    with open('valid.stamps', 'wb') as f:
        pickle.dump(valid, f)
    with open('train.stamps', 'wb') as f:
        pickle.dump(train, f)
    return test, valid, train

if __name__ == '__main__':
    before = os.getcwd()
    os.chdir('data')
    print('Creating directories')
    create_dirs()
    print('Unpacking tars')
    unpack_all_tars()
    print('Simplifying names')
    simplify_all_names()
    print('Removing images without masks')
    remove_images_without_matching_masks()
    print('Simplifying images')
    print(str(simplify_images()) + ' images processed')
    print('Simplifying masks')    
    print('[Blue, White, Black] = ' + str(simplify_masks()))
    print('Separating data') 
    test, valid, train = separate_data()
    print(str(len(test)) + ' test cases; ' + 
          str(len(valid)) + ' validation cases; ' + 
          str(len(train)) + ' training cases.')
    os.chdir(before)
    print('Done')
    