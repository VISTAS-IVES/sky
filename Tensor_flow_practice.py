#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson
"""

import tensorflow as tf
    
if __name__ == '__main__':
    inputs = [[0,0,255],[0,255,0], [0, 255, 255]]
    correct = [[1, 0, 0], [0,1,0], [0, 0, 1]]
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
