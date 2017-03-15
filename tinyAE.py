# -*- coding: utf-8 -*-
"""A  Cloud  Auto-encoder Network:
    One hidden layer, fully connected, Gradient Descent,
    cost function - mean squared error
@author: drake (borrowing heavily from gorr)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils

tf.reset_default_graph()

# Reads in image and flattens it out
x = utils.read_inputs_flat("data/filteredImages0518")

# Number of hidden units
n_hidden = 1

# Number of features
n_features = x.shape[1]


def layer(previous, number_of_outputs):
    """Create a network and return its output node."""
    dims = [int(previous.get_shape()[1]), number_of_outputs]
    w = tf.Variable(tf.random_uniform(dims, -1, 1))
    b = tf.Variable(tf.zeros([number_of_outputs]))
    return tf.nn.relu(tf.matmul(previous, w) + b)

# Build graph
tf.reset_default_graph()
train_in = tf.placeholder(tf.float32, shape=[None, n_features])
a1 = layer(train_in, n_hidden)
a2 = layer(a1, n_features)
cost = tf.reduce_mean(tf.squared_difference(a2, train_in), 1)
cost = tf.reduce_mean(cost)  # across batches
learning_rate = 1.0e-6
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Let's look at one training image
ind = 100
example = x[ind]
ex_image = example.reshape(480, 480, 3)
plt.figure()
plt.title("Original Image")
plt.imshow(ex_image)

# Train the network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

test_example = x[ind:ind+1, :]
n_iter = 1000
print('iter\ttrain_cost')
for i in range(n_iter):
    batch = utils.next_batch(x, 50)
    sess.run(optimizer, feed_dict={train_in: batch})
    if (i % (int(n_iter / 10)) == 0):
        batch_cost = sess.run(cost, feed_dict={train_in: batch})
        print(i, '\t', batch_cost)
#    if (i%(int(n_iter/100))==0):
#        recon = sess.run(Y, feed_dict={X: test_example})
#        test_image = recon.reshape(480,480,3)
#        plt.figure()
#        plt.title("iteration "+ str(i))
#        plt.imshow(np.clip(test_image/255., 0, 1))

ex = example.reshape((-1, 691200))
recon = sess.run(a2, feed_dict={train_in: ex})

recon_images = recon.reshape((480, 480, 3))
plt.figure()
plt.title("final")
plt.imshow(np.clip(recon_images/255., 0, 1))
