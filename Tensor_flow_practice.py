#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson
"""

import tensorflow as tf
    
if __name__ == '__main__':
#    inputs = [[0,0,255],[255,255,255],[0,0,0],[0,0,255]]
#    correct = [[0,0,1], [1,1,1], [0,0,0], [0,0,1]]
    inputs = [[0,0,255],[0,255,0], [0, 255, 255]]
    correct = [[0,0,1], [0,1,0], [0, 0.5, 0.5]]
    # if blue out is 0,0,1 if white 1,0,0 if black 0,1,0
    # Define the network
    x = tf.placeholder(tf.float32, [None, 3])
    y = tf.nn.softmax(x)
    y_ = tf.placeholder(tf.float32, [None, 3])
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=x))
    # Run it
    sess = tf.Session()
    print(sess.run(y, feed_dict={x: inputs}))
    print(sess.run(cross_entropy, feed_dict={x: inputs, y_: correct}))
