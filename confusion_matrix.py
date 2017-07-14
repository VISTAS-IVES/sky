"""
Created on Thu July 13 2017

@author: jeffmullins
"""

from train import build_net, get_inputs, get_nsmasks
from show_output import out_to_image, read_parameters, read_last_iteration_number
from analyze import read_valid_stamps, read_target
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
from scipy import misc
import pickle
import matplotlib.pyplot as plt
import argparse

BATCH_SIZE = 10

BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])
GRAY = np.array([192, 192, 192])
BLACK = np.array([0, 0, 0])

def update_confusion_matrix(matrix, result, correct):
    colors = [BLUE, WHITE, GRAY, BLACK]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i][j] += np.sum(np.logical_and((result == colors[j]).all(axis=2), (correct == colors[i]).all(axis=2)))
    

def make_confusion_matrix(num_worst, time_stamps, directory, step_version, kernel, layers, num):
    """Returns the the timestamps of the num_worst images for which the network
    most disagrees with the target masks."""
    
    confusion_matrix = [[0.0]*num for _ in range(num)]
    train_step, accuracy, saver, init, x, y, y_, cross_entropy = build_net(layers)
    with tf.Session() as sess:
        saver.restore(sess, directory + 'weights-' + str(step_version))
        for i, s in enumerate(time_stamps):
            inputs = get_inputs([s])
            result = out_to_image(y.eval(feed_dict={x: inputs}))
            result = result.reshape(480, 480, 3)
            correct = read_target(s)
            update_confusion_matrix(confusion_matrix, result, correct)
        confusion_matrix = np.asarray(confusion_matrix)
        print (confusion_matrix)
        for i in range(len(confusion_matrix)):
            print (confusion_matrix[i])
            print (np.sum(confusion_matrix[i]))
            print (100*np.sum(confusion_matrix[i]))
            confusion_matrix[i] = confusion_matrix[i]*100/(np.sum(confusion_matrix[i]))
        print(confusion_matrix)

if __name__ == '__main__':
    num = 3
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--black', action='store_true')
    args = parser.parse_args()
    if args.black:
        num = 4
    timestamps = read_valid_stamps(BATCH_SIZE)
    dir_name = "results/" + args.directory + "/"       
    args = read_parameters(dir_name)
    step_version = read_last_iteration_number(dir_name)
    layer_info = args['Layer info'].split()
    
    make_confusion_matrix(5, timestamps, dir_name, step_version, kernel_width, layer_info, num)