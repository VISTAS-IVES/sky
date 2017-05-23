#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson
"""
import numpy as np
from scipy import misc
import os
from PIL import Image
import tensorflow as tf
    
    
def mask_to_one_hot(img):
    BLACK = np.array([0, 0, 0])
    BLUE = np.array([0, 0, 255])
    WHITE = np.array([255, 255, 255])

    img[(img == WHITE).all(axis = 2)] = [1,0,0]
    img[(img == BLUE).all(axis = 2)]  = [0,1,0]
    img[(img == BLACK).all(axis = 2)] = [0,0,1]
    return img
  

if __name__ == '__main__':
    #get image and make the mask into a one-hotted mask
    inputs = misc.imread('data/images/sgptsiskyimageC1.a1.20160414.162830.jpg.20160414162830.jpg')
    correct = mask_to_one_hot(misc.imread('data/simplified_masks/sgptsicldmaskC1.a1.20160414.162830.png.20160414162830.png'))

    
    #flatten the images and masks
    inputs = inputs.reshape(-1,3)
    correct = correct.reshape(-1,3)
    
    
    # if blue out is 0,0,1 if white 1,0,0 if black 0,1,0
    # Define the network
    x = tf.placeholder(tf.float32, [None, 3])
    W = tf.Variable(tf.zeros([3, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 3])
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # Train
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for _ in range(1000):
        sess.run(train_step, feed_dict={x: inputs, y_: correct})
    # Test
    print(sess.run(y, feed_dict={x: inputs}))
