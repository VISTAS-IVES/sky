#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:58:47 2017

@author: drake
"""

from net import build_net, get_inputs, WHITE, BLUE, BLACK
import numpy as np
import tensorflow as tf
from PIL import Image

def one_hot_to_mask(max_indexs, output):
    """Modifies (and returns) img to have sensible colors in place of
    one-hot vectors."""
    output[(max_indexs == 0)] = WHITE
    output[(max_indexs == 1)] = BLUE
    output[(max_indexs == 2)] = BLACK
    return output

def out_to_image(output, n):
    """Modifies (and returns) the output of the network for the nth image as a
    human-readable RGB image."""
    output = output.reshape([-1,480,480,3])[n]
    outs = output
    # We use argmax instead of softmax so that we really will get one-hots
    max_indexes = np.argmax(outs, axis = 2)
    return one_hot_to_mask(max_indexes, outs)

def load_net(train_step, accuracy, saver, init, x, y, y_, num, result_dir):
     # Train
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num))
        inputs = get_inputs([20160414162830])
        img = out_to_image(y.eval(feed_dict={x: inputs}), 0)
        img = Image.fromarray(img.astype('uint8'))
        img.show()
#        img.save('results/out2.png')

if __name__ == '__main__':
   load_net(*build_net(), 5000, 'results/test3/')
