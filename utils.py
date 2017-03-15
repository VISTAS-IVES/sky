# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:40:51 2017

@author: gorr
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_inputs(inFolder):
    #inFolder = "filteredImages"
    inputNames = [f for f in listdir(inFolder) if isfile(join(inFolder, f))]
    inputs = np.zeros((len(inputNames),480,480,3),dtype=np.uint8)
    print('inputs',inputs.shape)
    
    for i in range(len(inputNames)):
        name = inFolder + '/' + inputNames[i]
        print(i, name, end=" ")
        im = mpimg.imread(inFolder + '/' + inputNames[i])
        inputs[i] = im
    return inputs

# Read in images and flatten each one
def read_inputs_flat(inFolder):
    inputNames = [f for f in listdir(inFolder) if isfile(join(inFolder, f))]
    inputs = np.zeros((len(inputNames),480*480*3),dtype=np.uint8)
    print('inputs',inputs.shape)
    
    for i in range(len(inputNames)):
        name = inFolder + '/' + inputNames[i]
        print(i, name)
        im = mpimg.imread(inFolder + '/' + inputNames[i])
        inputs[i] = im.flatten()
    return inputs


def next_batch(x, num):
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    b = x[perm[0:num]]
    return b
    
def showInput(im):
    plt.figure()    
    plt.title("image")
    plt.imshow(im)
    plt.show()


