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

# 
def read_inputs_outputs(inputFolder,targetFolder):
#    inputFolder = '../filteredImages0518'
#    targetFolder = '../mask0518'
    inputNames = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
    inputs = np.zeros((len(inputNames),480,480,3),dtype=np.uint8)
    targets = np.zeros((len(inputNames),480,480,3),dtype=np.uint8)
    
    
    for i in range(len(inputNames)):
        inputName = inputFolder + '/' + inputNames[i]
        targetName = targetFolder + '/mask'+ inputNames[i][8:24]+'.png'
        im  = mpimg.imread(inputName)
        imt = mpimg.imread(targetName)
        inputs[i] = im
        targets[i]= np.uint8(imt*255)
    print('inputs',inputs.shape,'targets',targets.shape )
    return [inputs,targets]


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

def next_batch(x, y, num):
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    xs = x[perm[0:num]]
    ys = y[perm[0:num]]
    return [xs,ys]
    
def showInput(im):
    plt.figure()    
    plt.title("image")
    plt.imshow(im)
    plt.show()
    
    


