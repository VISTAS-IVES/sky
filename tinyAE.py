# -*- coding: utf-8 -*-
"""A  Cloud  Auto-encoder Network:
    One hidden layer, fully connected, Gradient Descent,
    cost function - mean squared error
@author: drake (borrowing heavily from gorr)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import utils
from time import time

# Read in images and flatten them out
data = utils.read_inputs_flat("data/filteredImages0518")

# Number of hidden units
n_hidden = 100

# Number of input/output features (pixels * color channels)
n_features = data.shape[1]

# Network learning rate
learning_rate = 1.0e-3


def layer(previous, number_of_outputs):
    """Create a sigmoid layer and return its output node."""
    dims = [int(previous.get_shape()[1]), number_of_outputs]
    w = tf.Variable(tf.random_uniform(dims, -1, 1))
    b = tf.Variable(tf.zeros([number_of_outputs]))
    return tf.nn.sigmoid(tf.matmul(previous, w) + b)


def display(image, title):
    """Displays a sky image (original or network output)."""
    plt.title(title)
    plt.imshow(image.reshape(480, 480, 3))
    plt.show()

# Build graph
tf.reset_default_graph()
input_layer = tf.placeholder(tf.float32, shape=[None, n_features])
hidden_layer = layer(input_layer, n_hidden)
output_layer = layer(hidden_layer, n_features)
cost = tf.reduce_mean(tf.squared_difference(output_layer * 255, input_layer))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Look at one training image
ind = 100
example = data[ind]
#display(example, 'Example input')

# Train the network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

test_example = data[ind:ind+1, :]
n_iter = 1000
print('iter\ttrain_cost')
before = time()
for i in range(1, n_iter + 1):
    batch = utils.next_batch(data, 50)
    sess.run(optimizer, feed_dict={input_layer: batch})
    if (i % (int(n_iter / 100)) == 0):
        # Print batch cost
        batch_cost = sess.run(cost, feed_dict={input_layer: batch})
        print(i, '\t', batch_cost)
        if (i % (int(n_iter / 10)) == 0):
            # Display example output
            ex = example.reshape((-1, n_features))
            ex_out = sess.run(output_layer, feed_dict={input_layer: ex})
#            display(ex_out, 'Example output')
after = time()
print('Elapsed time: {}'.format(after - before))


def explore(hidden):
    """Displays the output image for the given hidden unit vector."""
    ex_out = sess.run(output_layer, feed_dict={hidden_layer: [hidden]})
    display(ex_out, 'Hidden layer: {}'.format(hidden))
