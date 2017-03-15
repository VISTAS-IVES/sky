# -*- coding: utf-8 -*-
"""A  Cloud  Auto-encoder Network:
    Multiple Layers, fully connected, Gradient Descent, 
    cost function - mean squared error
@author: gorr 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

# Reads in image and flattens it out
x = utils.read_inputs_flat("data/filteredImages0518")
#%%  Cell separator

tf.reset_default_graph()

# The encoding dimensions of our network. Note, for the auto-encoder,
# reconstruct the original image using decoding layers made up of 
# the same weights as in the encoding part but 
# transposed and in reverse order.  
dimensions = [10,10]  # encoding layers 

n_features = x.shape[1] 

X = tf.placeholder(tf.float32, [None, n_features],name='X')

# let's first copy our X placeholder to the name current_input
current_input = X
n_input = n_features

# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []

# We'll create a for loop to create each layer:
for layer_i, n_output in enumerate(dimensions):
    with tf.variable_scope("encoder/layer/{}".format(layer_i)):
        # Create a weight matrix which will increasingly reduce
        # down the amount of information 
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        h = tf.matmul(current_input, W)
        current_input = tf.nn.relu(h)
        # Finally we'll store the weight matrix so we can build the decoder.
        Ws.append(W)
        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output

# We'll first reverse the order of our weight matrices
Ws = Ws[::-1]

# then reverse the order of our dimensions appending the last layers number of inputs.
dimensions = dimensions[::-1][1:] + [x.shape[1]]
print(dimensions)

Wsd=[]
for layer_i, n_output in enumerate(dimensions):
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):

        # Now we'll grab the weight matrix we created before and transpose it
        # So a 3072 x 784 matrix would become 784 x 3072
        # or a 256 x 64 matrix, would become 64 x 256
        W = tf.transpose(Ws[layer_i])
        Wsd.append(W)

        h = tf.matmul(current_input, W)
        current_input = tf.nn.relu(h)

        # We replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output
 
Y = current_input
cost = tf.reduce_mean(tf.squared_difference(X, Y), 1) # across pixels
cost = tf.reduce_mean(cost)    # across batches
learning_rate = 0.001  
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#%%  Cell separator
#Let's look at one training image
ind = 100
example = x[ind] 
ex_image = example.reshape(480,480,3)
plt.figure()
plt.title("Original Image")
plt.imshow(ex_image)

#%%  Cell separator
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#%%  Cell separator

test_example = x[ind:ind+1,:]  
n_iter = 1000;
print('iter\ttrain_cost')
for i in range(n_iter):
    batch = utils.next_batch(x, 50)
    sess.run(optimizer, feed_dict={X: batch}) 
    if (i%(int(n_iter/10))==0):
        batch_cost = sess.run(cost, feed_dict={X: batch})  
        print(i, '\t',batch_cost)
    if (i%(int(n_iter/100))==0):
        recon = sess.run(Y, feed_dict={X: test_example})
        test_image = recon.reshape(480,480,3)
        plt.figure()
        plt.title("iteration "+ str(i))
        plt.imshow(np.clip(test_image/255., 0, 1))

#This takes too long if large number of inputs:
#all_recon = sess.run(cost, feed_dict={X: x})  
#print("cost over all examples:",all_recon)
#%%  Cell separator 
 
ex = example.reshape((-1, 691200))
recon = sess.run(Y, feed_dict={X: ex})

recon_images = recon.reshape((480,480,3))
plt.figure()
plt.title("final")
plt.imshow(np.clip(recon_images/255., 0, 1))
    
