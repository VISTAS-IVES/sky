# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:07:48 2017
We train a MLP network to classify a cloud image.
Target vectors represent the fraction of each color found in the corresponding 
mask file as follows:
targets[0]:  black - background  <---   Here, we delete this and rescale!
targets[1]:  blue - clear sky   
targets[2]:  white - thick clouds
targets[3]:  gray - wispy clouds
targets[4]:  green  - region markers 
targets[5]:  other  - this generally was the sun area which was yellow when 
                      the sky is clear but could also be grayish (could this 
                      accidently fall under the wispy category?)

The network consists of two convolutional layers followed by two 
fully connected layers:
    
input layer: 480x480xchan_input -  this is shape of rgb image
first convo layer:  
    filter shape (filter_size, filter_size, chan_input,chan0)
    480x480x3 -> 480x480xchan0

    max pool layer with stride 2,  
    480x480xchan0 -> 240x240xchan0
    
second convo layer: 
    filter shape (filter_size, filter_size, chan0, chan1)
    240x240xchan0 -> 240x240xchan1
    max pool layer with stride 2,  
    240x240xchan1 -> 120x120xchan1

fully connected layer
    120x120xchan1 -> nClasses

dropout layer 
   
fully connected layer
    120x120xchan1 -> nClasses
@author: gorr
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import montage

checkptSave = True  # set to True if you want to save to checkpoint
checkptRestore = False # set to True to restore from checkpoint (make sure checkptName is right)
folder = 'checkPoints/'
checkptName = folder + "cloud_class5meanShift.ckpt"  #Name of checkpoint file to restore
checkptNameRestore = folder + "cloud_class5meanShift.ckpt"  #Name of checkpoint file to save to

x0 = np.load('../cloudInputs0518.npy');  # cloud image
y6_classes = np.load('../cloudTargets0518.npy');  # fractions as described above

#%%  Cell separator
# This takes a long time!
# Preprocessing the inputs as follows:
# compute per pixel mean, std over all the images. Shift the inputs by the mean and divide by std.
# Note, the black pixels are all zero so the std is also zero.  This causes divide by zero problem
# so I replace all zero std values with 1. Thus, when one divides (0/1) one just get zero back.
x_mean = np.mean(x0,axis=0) 
x_std = np.std(x0,axis=0)
indices = np.where(x_std == 0) # locate indices of 0 std's
x_std[indices] = 1  # replace 0 with 1 to avoid divide by zero
x = (x0-x_mean)/x_std

xdims = np.shape(x)
#%%  Cell separator
#Let's remove the black data (column 0) from target classes and then rescale
y5 = np.delete(y6_classes,0, axis=1)  # delete 0th column
row_sums = np.sum(y5, axis=1)    # get row sums
y = y5/row_sums[:,None]     # rescale

chan_input = 3  # this is the 3 RGB color channels  
chan0 = 32  # number of channels in convo hidden layer. 
chan1 = 64 
filter_size = 5 # Convo filter size for hidden layers
nNodes = 50  # Number of nodes in first fully connected layer
nClasses = len(y[0])  # Number of nodes in second fully connected 
                      # layer. This is output layer. nClasses also
                      # represents the number of output classes.
drop_prob = 0.5  # dropout probability
learning_rate = 0.001   
batch_size = 20

#np.set_printoptions(threshold=np.inf) 

#%%  Cell separator
# ********  Helper Functions **************
def weight_variable(shape, vname):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=vname)

def bias_variable(shape, vname):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=vname)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x,vname):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME',name=vname)
 # ******************************************* 
 #%%  Cell separator
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 480,480,chan_input],name='X')
Y = tf.placeholder(tf.float32, [None, nClasses],name='Y')

# First Convolutional Layer
with tf.name_scope('convo_layer1'):
    W_conv1 = weight_variable([filter_size, filter_size, 3, chan0], 'W_conv1')
    b_conv1 = bias_variable([chan0],'b_conv1')
    h_conv1 = tf.nn.softplus(conv2d(X, W_conv1) + b_conv1, name='h_conv1')
    h_pool1 = max_pool_2x2(h_conv1,'h_pool1')
    
# note, this is size of h_pool1 (assuming max_pool stride=2)
h1_shape= [int(np.ceil(xdims[1]/2)),int(np.ceil(xdims[2]/2)), chan0]
print('h1_shape',h1_shape)
    
# Second Convolutional Layer
with tf.name_scope('convo_layer2'):
    W_conv2 = weight_variable([filter_size, filter_size, chan0, chan1],'W_conv2')
    b_conv2 = bias_variable([chan1],'b_conv2')
    h_conv2 = tf.nn.softplus(conv2d(h_pool1, W_conv2) + b_conv2, name='h_conv2')
    h_pool2 = max_pool_2x2(h_conv2,'h_pool2')

# Densely Connected Layers

#Need to get shape of h_pool2 layer - not sure how else to get it 
# note, this is size of h_pool2 (assuming max_pool stride=2)
h2_shape= [int(np.ceil(h1_shape[0]/2)),int(np.ceil(h1_shape[1]/2)), chan1]
print('h2_shape',h2_shape)

with tf.name_scope('fullyConnected_layer'):
    n_features = h2_shape[0] * h2_shape[1] * chan1
    W_fc1 = weight_variable([n_features, nNodes],'W_fc1')
    b_fc1 = bias_variable([nNodes],'b_fc1')

    h_pool2_flat = tf.reshape(h_pool2, [-1, n_features])
    h_fc1 = tf.nn.softplus(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name='h_fc1')

# Dropout layer
with tf.name_scope('dropout_layer'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name='h_fc1_drop')

# Output Layer
with tf.name_scope('output_layer'):
    W_fc2 = weight_variable([nNodes, nClasses],'W_fc2')
    b_fc2 = bias_variable([nClasses],'b_fc2')
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=Y),name='cross_entropy')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(Y,1),name='correct_prediction')
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

#%%  Cell separator
# Initialize Tensorflow session

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#If you want to restore from saved file:
if checkptRestore:
    saver = tf.train.Saver()
    saver.restore(sess, checkptNameRestore)
    print('Restoring from ',checkptNameRestore)

#%%  Cell separator
# Train Network

n_iter = 101
iter_start = 0
print("iter\tcost")
for i in range(n_iter):
    # randomly select elements
    batch = utils.next_batch(x,y, batch_size)
    sess.run(train_step, feed_dict={X: batch[0], Y:batch[1],keep_prob: drop_prob })
    
    if (i+iter_start) % 2 == 0:
        cost = sess.run(cross_entropy, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
        print(i+iter_start, '\t',cost)

        # check to see if outputs of 1st fully connected nodes are all zero
        h_fc1c = sess.run(h_fc1, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
        print('h_fc1c',np.sum(h_fc1c), np.min(h_fc1c), np.max(h_fc1c))

#        h_fc1_dropc = sess.run(h_fc1_drop, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
#        print('h_fc1_dropc', np.sum(h_fc1_dropc), np.min(h_fc1_dropc), np.max(h_fc1_dropc))

    if (i+iter_start)%50==49 and checkptSave:
        saver = tf.train.Saver()
        save_path = saver.save(sess, checkptName, global_step=i+iter_start)
        print("Model saved:", save_path)    

#%%  Cell separator
# Look at output of various layers to see where vanishing ReLU problem occurs:
# Layers: X->h_conv1->h_pool1->h_conv2->(h_pool2/h_pool2_flat)->h_fc1->h_fc1_drop->y_conv

batch = utils.next_batch(x,y, 5)  # look at subset of all inputs

#W_fc1c = sess.run(W_fc1)
#print('W_fc1c', np.sum(W_fc1c), 'min:',np.min(W_fc1c), 'max:',np.max(W_fc1c))
#
#b_fc1c=sess.run(b_fc1, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
#print('b_fc1c', b_fc1c)

h_conv1c =  sess.run(h_conv1, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_conv1', np.sum(h_conv1c), 'min:',np.min(h_conv1c), 'max:',np.max(h_conv1c))

h_pool1c =  sess.run(h_pool1, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_pool1c', np.sum(h_pool1c), 'min:',np.min(h_pool1c), 'max:',np.max(h_pool1c))

h_conv2c =  sess.run(h_conv2, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_conv1', np.sum(h_conv2c), 'min:',np.min(h_conv2c), 'max:',np.max(h_conv2c))

h_pool2c =  sess.run(h_pool2, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_pool2c', np.sum(h_pool2c), 'min:',np.min(h_pool2c), 'max:',np.max(h_pool2c))

#h_pool2_flatc = sess.run(h_pool2_flat, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
#print('h_pool2_flatc', np.sum(h_pool2_flatc), 'min:',np.min(h_pool2_flatc), 'max:',np.max(h_pool2_flatc) )

temp=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
tempc = sess.run(temp, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_pool2_flat, W_fc1) + b_fc1', np.sum(tempc), np.max(tempc), np.min(tempc))

h_fc1c = sess.run(h_fc1, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_fc1c',np.sum(h_fc1c), np.min(h_fc1c), np.max(h_fc1c))

h_fc1_dropc = sess.run(h_fc1_drop, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('h_fc1_dropc', np.sum(h_fc1_dropc), np.min(h_fc1_dropc), np.max(h_fc1_dropc))

cost = sess.run(cross_entropy, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
#y_pred = sess.run(tf.nn.softmax(y_conv), feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
y_pred = sess.run(y_conv, feed_dict={X: batch[0], Y:batch[1],keep_prob: 1.0})
print('batch\n',batch[1])
print('cost:',cost)
print(y_pred)
for i in range(5): 
    y_pred = sess.run(tf.nn.softmax(y_conv), feed_dict={X: batch[0][i], Y:batch[1][i],keep_prob: 1.0})
    print('targets:\n',batch[1][i])
    print('predicted:\n',y_pred[i],'\n')

# Look at weight filters in first layer
W1 = sess.run(W_conv1)
b1 = sess.run(b_conv1)

# Look at weight filters in layer 1 
m1 = montage.montage_filters(W1)
plt.figure()
plt.axis('off')
#plt.imshow(m1, cmap = 'seismic', interpolation='nearest') 
#plt.imshow(m1, cmap = 'seismic', interpolation='bicubic') 
plt.imshow(m1, cmap = 'gray', interpolation='bicubic') 

# Look at weight filters in layer 2 
W2 = sess.run(W_conv2)
m2 = montage.montage_filters(W2)
plt.figure()   
plt.imshow(m2, cmap = 'seismic',interpolation='bicubic')  

