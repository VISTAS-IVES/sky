"""
Created on Thu July 13 2017

@author: jeffmullins
"""

from train import build_net, get_inputs, format_nsmask, get_nsmasks, get_masks
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

BATCH_SIZE = 50

BLUE = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])
GRAY = np.array([192, 192, 192])
BLACK = np.array([0, 0, 0])

TIME_STAMPS = read_valid_stamps(BATCH_SIZE)
print (TIME_STAMPS)

def read_targets():
    """Reads and returns the target mask corresponding to timestamps from
    the simplemask directory."""
    ret = np.zeros((len(TIME_STAMPS),480,480,3))
    for i in range(len(TIME_STAMPS)):
        ret[i] = np.array(misc.imread('data/simplemask/simplemask' + str(TIME_STAMPS[i]) + '.png'))
    return ret

def update_confusion_matrix(matrix, result, correct):
    colors = [BLUE, WHITE, GRAY, BLACK]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i][j] += np.sum(np.logical_and((result == colors[j]).all(axis=2), (correct == colors[i]).all(axis=2)))
    
def run_net(train_step, accuracy, saver, init, x, y, y_, cross_entropy, result_dir, num_iterations):
    """Loads and runs the most recent network from result_dir. Returns the input impage and the output one-hot mask."""
    with tf.Session() as sess:
        saver.restore(sess, result_dir + 'weights-' + str(num_iterations))
        inputs = get_inputs(TIME_STAMPS)
        result = y.eval(feed_dict={x: inputs})
        result.reshape(-1, 480, 480, 5)
        accuracy = accuracy.eval(feed_dict={x: inputs, y_: get_masks(TIME_STAMPS)})
        print('Accuracy = ' + str(accuracy))
        return inputs, result
    
def load_net(train_step, accuracy, saver, init, x, y, y_, cross_entropy, result_dir, num_iterations):
    inputs, result = run_net(train_step, accuracy, saver, init, x, y, y_, cross_entropy, result_dir, num_iterations)
    return result


def make_ensemble_matrix(counts,time_stamps):
    confusion_matrix = [[0.0]*4 for _ in range(4)]
    print (np.shape(confusion_matrix))
    results = out_to_image(counts)
    for i, s in enumerate(time_stamps):
        correct = read_target(s)
        result = results[i]
        update_confusion_matrix(confusion_matrix, result, correct)
    confusion_matrix = np.asarray(confusion_matrix)
    print (confusion_matrix)
    for i in range(len(confusion_matrix)):
        print (confusion_matrix[i])
        print (np.sum(confusion_matrix[i]))
        print (100*np.sum(confusion_matrix[i]))
        confusion_matrix[i] = confusion_matrix[i]*100/(np.sum(confusion_matrix[i]))
    print(confusion_matrix)
    
def find_accuracy(counts,time_stamps):
    results = out_to_image(counts)
    corrects = read_targets()
    return np.sum((results == corrects).all(axis=3))/(len(TIME_STAMPS)*480*480)

#def make_confusion_matrix(num_worst, time_stamps, directory, step_version, kernel, layers, num):
#    """Returns the the timestamps of the num_worst images for which the network
#    most disagrees with the target masks."""
#    
#    confusion_matrix = [[0.0]*num for _ in range(num)]
#    for i, s in enumerate(time_stamps):
#        inputs = get_inputs([s])
#        result = out_to_image(y.eval(feed_dict={x: inputs}))
#        result = result.reshape(480, 480, 3)
#        correct = read_target(s)
#        update_confusion_matrix(confusion_matrix, result, correct)
#    confusion_matrix = np.asarray(confusion_matrix)
#    print (confusion_matrix)
#    for i in range(len(confusion_matrix)):
#        print (confusion_matrix[i])
#        print (np.sum(confusion_matrix[i]))
#        print (100*np.sum(confusion_matrix[i]))
#        confusion_matrix[i] = confusion_matrix[i]*100/(np.sum(confusion_matrix[i]))
#    print(confusion_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='+')
    args = parser.parse_args()
    print (args.directory)
    
    counts = np.zeros([len(TIME_STAMPS),480,480, 5])
    for d in args.directory:
        dir_name = d + "/" # Does NOT include results/ to allow shell globbing
        print(dir_name)
        args = read_parameters(dir_name)
        step_version = read_last_iteration_number(dir_name)
        layer_info = args['Layer info'].split()
        onehot = load_net(*build_net(layer_info, 0), dir_name, step_version)
        onehot = onehot.reshape((len(TIME_STAMPS),480,480,5))
        counts += onehot
        
    print ("ENSEMBLE ACCURACY:" + str(find_accuracy(counts,TIME_STAMPS)))

#    num = 3
#    parser = argparse.ArgumentParser()
#    parser.add_argument('directory')
#    parser.add_argument('--black', action='store_true')
#    args = parser.parse_args()
#    if args.black:
#        num = 4
#    timestamps = read_valid_stamps(BATCH_SIZE)
#    dir_name = "results/" + args.directory + "/"       
#    args = read_parameters(dir_name)
#    step_version = read_last_iteration_number(dir_name)
#    layer_info = args['Layer info'].split()
#    
#    make_confusion_matrix(5, timestamps, dir_name, step_version, kernel_width, layer_info, num)
