#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:03:58 2017

@author: jeffmullins
"""
from train import build_net
from show_output import read_parameters, read_last_iteration_number
import numpy as np
import tensorflow as tf
import math
import sys
from PIL import Image


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
    
    #put all filters onto one image with a layer of black between each
    big_img = np.zeros((math.ceil(layer_size/8)*(kernel_size+1)+1,(kernel_size+1)*8+1,3))
    for i in range(layer_size):
        y = (i//8)*(kernel_size+1)+1
        x = (i%8)*(kernel_size+1)+1
        for r in range(kernel_size):
            for c in range(kernel_size):
                big_img[r+y][c+x] = reshaped_kernel[i][r][c]
    #show the filters
    img = Image.fromarray(big_img.astype('uint8'))
    img.show()


def load_net(train_step, accuracy, saver, init, x, y, y_, ns, cross_entropy, result_dir, num_iterations, kernel_width, layer_size):
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num_iterations))
        with tf.variable_scope('hidden0'):
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights')
            put_kernels_on_grid (weights.eval(), kernel_width, layer_size)
            
            
if __name__ == '__main__':
    dir_name = sys.argv[1]
    dir_name = "results/" + dir_name + "/"
    args = read_parameters(dir_name)
    step_version = read_last_iteration_number(dir_name)
    kernel_width = int(args['Kernel width'])
   # layer_sizes = list(map(int, args['Layer sizes'].split()))
    
    layer_sizes = [32]
    load_net(*build_net(0, kernel_width, layer_sizes), dir_name, step_version, kernel_width, layer_sizes[0])
#    with tf.Session() as sess:
#       saver.restore(sess, directory + 'weights-' + str(step_version))
#       with tf.variable_scope('hidden0'):
#           tf.get_variable_scope().reuse_variables()
#           weights = tf.get_variable('weights')
#           put_kernels_on_grid (weights.eval())
           



