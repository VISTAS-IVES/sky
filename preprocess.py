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
import time

WHITE = np.array([255, 255, 255])
BLUE = np.array([0, 0, 255])
GRAY = np.array([192, 192, 192])
BLACK = np.array([0, 0, 0])
GREEN = np.array([0, 255, 0])
YELLOW = np.array([255, 255, 0])


def create_dirs():
    os.mkdir('skyimage')
    os.mkdir('cldmask')
    os.mkdir('simpleimage')
    os.mkdir('simplemask')
    os.mkdir('nsmask')


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
    for f in os.listdir('cldmask/'): # Was skyimage
        g = 'cldmask/cldmask' + extract_timestamp(f) + '.png'
        
        #
        h = 'skyimage' + extract_timestamp(f) + '.jpg'
#        if not os.path.isfile(g):
#            os.remove('skyimage/' + h) # Was f
        #elif os.path.getsize(g) == 0:
        if os.path.getsize(g) == 0:            
#            os.remove('skyimage/' + h) # Was f
            os.remove(g)


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
    """Returns an array of the number of BLUE, WHITE, and GRAY pixels
    in img."""
    blue = (img == BLUE).all(axis=2).sum()
    white = (img == WHITE).all(axis=2).sum()
    gray = (img == GRAY).all(axis=2).sum()
    return np.array([blue, white, gray])


def simplify_colors(img):
    """Returns an image with GREEN and YELLOW pixels made black. Destructively modifies img."""
    img[(img == GREEN).all(axis=2)] = BLACK
    img[(img == YELLOW).all(axis=2)] = BLACK
    return img


def depth_first_search(r, c, img, visited, ever_visited, stack):
    """Returns True if there is a connected region including img[r][c] that is all
    WHITE and surrounded by BLACK. Modifies visited to include all of the white pixels.
    Modified ever_visited to include all pixels explored."""
    while stack:
        r, c = stack.pop()
        if ((img[r][c] == BLACK).all()):
            continue
        if (visited[r][c]):
            continue
        visited[r][c] = True
        if (ever_visited[r][c]):
            return False
        ever_visited[r][c] = True
        if (img[r][c] == GREEN).all() or (img[r][c] == BLUE).all() or (img[r][c] == GRAY).all():
            return False
        stack.extend(((r+1, c), (r-1, c), (r, c+1), (r, c-1)))
    return True


def remove_white_sun(img):
    """Removes the sun disk from img if it is white. (A yellow sun is easier
    to remove; that is handled by simplify_colors.)"""
    start = time.clock()
    ever_visited = np.full(img.shape[:2], False, dtype=bool)
    visited = np.full(img.shape[:2], False, dtype=bool)
    for r in range(0, img.shape[0], 10):
        for c in range(0, img.shape[1], 10):
            if ((img[r][c] == WHITE).all()):
                stack = []
                stack.append((r, c))
                visited.fill(False)
                if depth_first_search(r, c, img, visited, ever_visited, stack):
                    img[visited] = BLACK
                    print('Removed the sun in ' + str(time.clock()-start) + ' seconds')
                    return img
    print('No sun found!')
    return img


def test_remove():
    img = misc.imread('data/cldmask/cldmask20160414174600.png')
    img = remove_white_sun(img)
    print(type(img))
    img = Image.fromarray(img.astype('uint8'))
    img.show()
    return img


def ns_mask_to_image(mask):
    """Takes a mask of booleans and returns an image where each pixel is
    BLACK if in the mask, BLUE otherwise."""
    img = np.full((480, 480, 3), BLUE)
    img[(mask)] = BLACK
    return img


def create_non_sky_mask(mask):
    """Given a mask, creates a saves an image that is BLACK
    for every place that is not sky and BLUE for every place that is sky"""
    ns_mask = np.full((480, 480, 3), BLUE)
    ns_mask[(mask == BLACK).all(axis=2)] = BLACK
    return ns_mask

def save_non_sky_masks():
    """Creates a mask of pixels which are black or green in every cldmask and
    saves it to always_black_mask.png."""
    for file in os.listdir('simplemask/'):
        mask = misc.imread('simplemask/' + file)
        ns_mask = create_non_sky_mask(mask)
        ns_mask = ns_mask.astype('uint8')
        Image.fromarray(ns_mask).save('nsmask/nsmask' + extract_timestamp(file) + '.png')

def simplify_masks():
    """Writes similified versions of all images in in_dir to out_dir.
    Returns an array of relative frequencies of WHITE, BLUE, and GRAY."""
    counts = np.zeros(3, dtype=np.int)
    for file in os.listdir('cldmask/'):
        img = misc.imread('cldmask/' + file)
        img = crop_image(img)
        print('About to remove sun from ' + file)
        if (img == YELLOW).all(axis=2).any():
            print('Sun is yellow')
        else:
            print('Removing white sun')
            img = remove_white_sun(img)
        simplified = simplify_colors(img)
        counts = counts + color_counts(simplified)
        Image.fromarray(simplified).save('simplemask/simplemask' + extract_timestamp(file) + '.png')
    return (counts / counts.sum())


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
#    print('Creating directories')
#    create_dirs()
#    print('Unpacking tars')
#    unpack_all_tars()
#    print('Simplifying names')
#    simplify_all_names()
    print('Removing images without masks')
    remove_images_without_matching_masks()
#   print('Simplifying images')
#   print(str(simplify_images()) + ' images processed')
    print('Simplifying masks')
    print('[White, Blue, Gray] = ' + str(simplify_masks()))
    print('Saving non-sky masks')
    save_non_sky_masks()
    print('Separating data')
    test, valid, train = separate_data()
    print(str(len(test)) + ' test cases; ' +
          str(len(valid)) + ' validation cases; ' +
          str(len(train)) + ' training cases.')
    os.chdir(before)
    print('Done')
