#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:20:00 2017

@author: Jeff Mullins, Sean Richardson
"""

import tensorflow as tf

def main():

    inputs = [[0,0,255],[255,255,255],[0,0,0],[0,0,255]]
    # if blue out is 0,0,1 if white 1,0,0 if black 0,1,0
    
    f = tf.constant(1.0/255.0)
    i = tf.placeholder(tf.float32, [4,3])
    o = tf.multiply(i, f)
    
    sess = tf.Session()
    result = sess.run(o, feed_dict={i: inputs})
    print (result)