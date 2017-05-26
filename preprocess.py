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

def unpack_tars():
    for f in os.listdir('./'):
        if f.endswith('.tar'):
            if 'skyimage' in f:
                unpack_tar(f, 'skyimage/')
            elif 'cldmask' in f:
                unpack_tar(f, 'cldmask/')
