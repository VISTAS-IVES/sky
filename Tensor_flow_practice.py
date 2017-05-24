#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson

Relies heavily on deep MNIST tutorial from TensorFlow.org.
"""
import numpy as np
from scipy import misc
import os
from PIL import Image
import tensorflow as tf
  
#define colors
BLACK = np.array([0, 0, 0])
BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])  
    
def mask_to_one_hot(img):
    img[(img == WHITE).all(axis = 2)] = np.array([1,0,0])
    img[(img == BLUE).all(axis = 2)]  = np.array([0,1,0])
    img[(img == BLACK).all(axis = 2)] = np.array([0,0,1])
    
    return img

def one_hot_to_mask(img):
    img[(img == np.array([1,0,0])).all(axis = 2)] = WHITE
    img[(img == np.array([0,1,0])).all(axis = 2)] = BLUE
    img[(img == np.array([0,0,1])).all(axis = 2)] = BLACK
    return img

def out_to_image(output):
    #takes the output of one-hot vectors and converts it into an RGB image
    output = output.reshape([480,480,3])
    output = one_hot_to_mask(output)
    return output

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

if __name__ == '__main__':
    # Get image and make the mask into a one-hotted mask
    inputs = misc.imread('data/simplified_images/sgptsiskyimageC1.a1.20160414.162830.jpg.20160414162830.jpg')
    print(inputs.shape)
    correct = mask_to_one_hot(misc.imread('data/simplified_masks/sgptsicldmaskC1.a1.20160414.162830.png.20160414162830.png'))
   # correct = misc.imread('data/simplified_masks/sgptsicldmaskC1.a1.20160414.162830.png.20160414162830.png')

    #Image.fromarray(correct, 'RGB').show()

    inputs = inputs.reshape(1, 480, 480, 3)
    correct = correct.reshape(-1, 3)
    # Define the network
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    W = weight_variable([3, 3, 3, 3])
    b = bias_variable([3])
    h = tf.nn.relu(conv2d(x, W) + b)
    y = tf.reshape(h, [-1, 3])
    y_ = tf.placeholder(tf.float32, [None, 3])
    out = tf.nn.softmax(y)
    
    # Is this minimizing across every output channel, pixel, and image?
    # Is that what we want?
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # Does this make sense? Does it count the number of pixels we got right?
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
#        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                    x:inputs, y_: correct})
#            train_accuracy = accuracy.eval(feed_dict={
#                    x:batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        sess.run(train_step, feed_dict={x: inputs, y_: correct})

    output =  (sess.run(out, feed_dict={x: inputs}))
    print (type(output))
    output = output.astype(int)
    output = out_to_image(output)
    print (type(output))
    #print (output.shape)
    #int_output = output.astype(int)
    #print (int_output)
    #print (np.where(output > 255))
    Image.fromarray(output.astype('uint8')).show()

    #Image.fromarray(output).show()
    
#        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    # Test    
#    accuracy = sess.run(accuracy, feed_dict={
#            x: mnist.test.images, y_: mnist.test.labels})
#    print("test accuracy %g"%accuracy)
#    print("test accuracy %g"%accuracy.eval(feed_dict={
#        x: mnist.test.images, y_: mnist.test.labels}))