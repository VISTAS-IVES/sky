#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:18:29 2017

@author: jeffmullins
"""

import numpy as np
from scipy import misc 
import math
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt


KERNEL = np.zeros((5,5,3,1))

for i in range(5):
    KERNEL[1][i][0][0] = i+1
    KERNEL[i][1][1][0] = i+6
    KERNEL[i][3][2][0] = 15-i

def put_kernels_on_grid (kernel, kernel_size, layer_size):
    kernel = np.array(kernel)
    reshaped_kernel = np.zeros((layer_size,kernel_size,kernel_size,3))
    for i in range(kernel_size):
        for n in range(kernel_size):
            for k in range(3):
                for r in range(layer_size):
                    reshaped_kernel[r][i][n][k] = kernel[i][n][k][r]
    #scale all values to be between 0 and 255
    for i in range(len(reshaped_kernel)):
        reshaped_kernel[i] = reshaped_kernel[i] - np.amin(reshaped_kernel[i]) 
        reshaped_kernel[i] = (255/np.amax(reshaped_kernel[i])) * reshaped_kernel[i]
    print(reshaped_kernel)
    #put all filters onto one image with a layer of black between each
    big_img = np.zeros((math.ceil(layer_size/8)*(kernel_size+1)+1,(kernel_size+1)+1,3))
    for i in range(layer_size):
        y = (i//8)*(kernel_size+1)+1
        x = (i%8)*(kernel_size+1)+1
        for r in range(kernel_size):
            for c in range(kernel_size):
                big_img[r+y][c+x] = reshaped_kernel[i][r][c]
    #show the filters
    img = Image.fromarray(big_img.astype('uint8'))
    img.show()

    
def conv2d(x, W, num):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = 'conv'+str(num))
    
def pretrained_convo_layer(prev):
    W = tf.constant(KERNEL, dtype = tf.float32)
    h = conv2d(prev, W, 0)   
    return h


def convo_layer(num_in, num_out, width, prev, num, relu=True):
    
    with tf.variable_scope('hidden' + str(num)):
        initial = tf.truncated_normal([width, width, num_in, num_out], stddev=(2 / math.sqrt(width * width * num_in)))
        W = tf.get_variable("weights", initializer = initial)
        initial = tf.constant(0.1, shape=[num_out])
        b = tf.Variable(initial, name='biases')
        if relu:
            h = tf.nn.relu(conv2d(prev, W, num) + b)
        else:
            h = conv2d(prev, W, num) + b   
    return h

    
def build_pretrained_net():
    print("Building network")
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    h = pretrained_convo_layer(x)
    y = tf.reshape(h, [-1, 1])    
    return x, y
    
def build_net():
    print("Building network")
    x = tf.placeholder(tf.float32, [None, 480, 480, 3])
    h = convo_layer(3, 1, 5, x, 0, relu=False)
    y = tf.reshape(h, [-1, 1])
    y_ = tf.placeholder(tf.float32, [None, 1])
    difference = y - y_
    loss = tf.nn.l2_loss(difference)
    train_step = tf.train.AdamOptimizer(1.0).minimize(loss)
    init = tf.global_variables_initializer()    
    return train_step, init, x, y, y_, loss

if __name__ == '__main__':
    put_kernels_on_grid(KERNEL, 5, 1)
    tf.reset_default_graph()
    x0, y0 = build_pretrained_net()
    inputs = np.random.randint(0, 256, (1, 480, 480, 3))
    train_step, init, x, y, y_, loss = build_net()
#    inputs = np.zeros((1, 480, 480, 3))
#    inputs[0,:,200,0] = 255
#    print(inputs[0,:,200,:])
    with tf.Session() as sess:
        correct_output = y0.eval(feed_dict={x0: inputs})
        img = correct_output.reshape([480, 480])
        plt.imshow(img, cmap = 'gray', vmin = np.amin(img), vmax = np.amax(img))
        plt.show
        init.run()
        for i in range(1000):
            train_step.run(feed_dict={x: inputs, y_: correct_output})
            print(loss.eval(feed_dict={x: inputs, y_: correct_output}))
        learned_output = y.eval(feed_dict={x: inputs})
        plt.imshow(img, cmap = 'gray', vmin = np.amin(img), vmax = np.amax(img))
        plt.show
        with tf.variable_scope('hidden0'):
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights')
            put_kernels_on_grid (weights.eval(), 5, 1)
