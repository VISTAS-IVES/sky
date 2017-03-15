# -*- coding: utf-8 -*-
"""A  Cloud Auto-encoder Network:
    Convolutional Neural Network (CNN), Gradient Descent, 
    cost function - mean squared error

@author: gorr 
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import montage
import datetime

checkptSave = False  # set to True if you want to save to checkpoint
checkptRestore = False # set to True to restore from checkpoint (make sure checkptName is right)
checkptName = "checkPoints/cnn_model.ckpt"  #Name of checkpoint file

x = utils.read_inputs("data/filteredImages0518")

#%%  Cell separator
# Build Tensorflow graph

tf.reset_default_graph()
n_features =  480*480*3
X = tf.placeholder(tf.float32, [None, 480,480,3],name='X')
X_flat = tf.reshape(X,[-1, n_features])

# let's first copy our X placeholder to the name current_input
current_input = X

# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []
shapes = []

# No pooling is used but because the stride is 2x2, the dimensions of each 
# channel is reduced by a half from the previous layer. That is, if we 
# have 10 and 5 channels in hidden layers, and we start with input layer
# 480x480x3, then the next layer will be 240x240x10, followed by
# 120x120x5 (this is the encoding), 
#followed by 240x240x10 and 480x480x3 (this is the decoding)
n_input = 3  # this is the 3 RGB color channels  
n_filters = [10, 5]  # number of channels in each hidden layer. 
filter_sizes = [10, 10] # convo filter sizes for each hidden layer

# We'll create a for loop to create each layer:
for layer_i, n_output in enumerate(n_filters):
    # just like in the last session,
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("encoder/layer/{}".format(layer_i)):
        # we'll keep track of the shapes of each layer
        # As we'll need these for the decoder
        shapes.append(current_input.get_shape().as_list())

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = tf.get_variable(
            name='W',
            shape=[
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input,
                n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        # Now we'll convolve our input by our newly created W matrix
        h = tf.nn.conv2d(current_input, W,
            strides=[1, 2, 2, 1], padding='SAME')
                
        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)

        # Finally we'll store the weight matrix so we can build the decoder.
        Ws.append(W)

        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output


# We'll first reverse the order of our weight matrices
Ws.reverse()
# and the shapes of each layer
shapes.reverse()
# and the number of filters (which is the same but could have been different)
n_filters.reverse()
# and append the last filter size which is our input image's number of channels
n_filters = n_filters[1:] + [1]

# and then loop through our convolution filters and get back our input image
# we'll enumerate the shapes list to get us there
for layer_i, shape in enumerate(shapes):
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = Ws[layer_i]
        
        # Now we'll convolve by the transpose of our previous convolution tensor
#        h = tf.nn.conv2d_transpose(current_input, W,
#            tf.pack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
#            strides=[1, 2, 2, 1], padding='SAME')
        h = tf.nn.conv2d_transpose(current_input, W,
            tf.stack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
            strides=[1, 2, 2, 1], padding='SAME')
 
        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)

Y = current_input
Y = tf.reshape(Y, [-1, n_features])


cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X_flat, Y), 1))
learning_rate = 0.001

# pass learning rate and cost to optimize
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#%%  Cell separator
#Let's look at a  training image
example = x[10]
plt.figure()
plt.title("Original Image")
plt.imshow(example)
plt.show()

#%%  Cell separator
# Initialize session

#sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

#If you want to restore from saved file:
if checkptRestore:
    saver = tf.train.Saver()
    saver.restore(sess, checkptName)
    print('Restoring from ',checkptName)

#%%  Cell separator
#Now let's train and look at reconstructed images 
startTime = datetime.datetime.now()
print("Start: date and time: " , startTime)
print('startTime',startTime)

print('iter\ttrain_cost')
n_iter = 201
for i in range(n_iter):
    batch = utils.next_batch(x, 50)
    sess.run(optimizer, feed_dict={X: batch})
    if i%20 == 0:   
        c_train = sess.run(cost, feed_dict={X: batch})
        print(i, '\t',c_train)
    if i%100 == 0: 
        recon = sess.run(Y, feed_dict={X: x[10:11]})
        re = np.reshape(recon,(480,480,3))
        plt.figure()
        plt.title("iter "+ str(i))
        plt.imshow(np.clip(re/255., 0, 1))
    if i%100==0 and checkptSave:
        saver = tf.train.Saver()
        save_path = saver.save(sess, checkptName, global_step=i)
        print("Model saved in file: %s" % save_path)

stopTime = datetime.datetime.now()
print("Start: date and time: " , startTime)
print("Stop: date and time: " , stopTime)
print("Time Diff: " , stopTime-startTime)

#all_recon = sess.run(cost, feed_dict={X: x})  
#print("cost over all examples:",all_recon)


## To save the graph to a checkpoint
#if checkptSave:
#    saver = tf.train.Saver()
#    save_path = saver.save(sess, checkptName)
#    print("Model saved in file: %s" % save_path)

#%%  Cell separator
# Let's look at some of the weights and activations


g = tf.get_default_graph()

# Let's look at one of the original images
j = 415
ex = x[j]
plt.figure()
plt.title("Original Image"+str(j))
plt.imshow(ex)
plt.show()

# Let's look at the corresponding reconstructed image
recon = sess.run(Y, feed_dict={X: x[j:j+1]})
re = np.reshape(recon,(480,480,3))
reScale = np.clip(re/255., 0, 1)   # make sure images are in range 0 to 255
plt.figure()
plt.title("recon scaled"+str(j))
plt.imshow(reScale)

# Let's grab the weights (these are independent of which image we are looking at)
exb = x[j:j+1]
W1 = g.get_tensor_by_name('encoder/layer/0/W:0')
W1c = sess.run(W1)
h1 = g.get_tensor_by_name('encoder/layer/0/Conv2D:0')
h1c = sess.run(h1,feed_dict={X:exb})

W2 = g.get_tensor_by_name('encoder/layer/1/W:0')
W2c = sess.run(W2)
h2 = g.get_tensor_by_name('encoder/layer/1/Conv2D:0')
h2c = sess.run(h2,feed_dict={X:exb})

# Let's plot some of the weights:
#  Weight filters in first layer
m1 = montage.montage_filters(W1c)
plt.figure()
plt.axis('off')
plt.title('AE CNN Layer 1 Weights')
plt.imshow(m1, cmap = 'seismic', interpolation='bicubic') 
#plt.imshow(m1, cmap = 'seismic', interpolation='nearest') 
plt.show()

# Weight filters in second layer
m2 = montage.montage_filters(W2c)
plt.figure()
plt.axis('off')
plt.title('AE CNN Layer 2 Weights')
plt.imshow(m2, cmap = 'seismic', interpolation='bicubic') 
#plt.imshow(m2, cmap = 'seismic', interpolation='nearest') 
plt.show()

# Let's plot some of the convolutions using the same image as above:
#Convolutions from layer 1 nodes
c = np.squeeze(h1c)
sh = np.rollaxis(c,2)
for k in range(sh.shape[0]):
    plt.figure()   
    plt.title('Layer 1 Convolutions for filter '+str(k)) 
    plt.axis('off')
    plt.imshow(sh[k])
    plt.show()


# Convolutions from layer 1 nodes
#sh = np.rollaxis(np.squeeze(h1c),2)
#shimage = montage.montage_images(sh)
#plt.figure()   
#plt.title('Layer 1 Convolutions') 
#plt.axis('off')
#plt.imshow(shimage)
#plt.show()
#
## Convolutions from layer 2 nodes
#sh = np.rollaxis(np.squeeze(h2c),2)
#shimage = montage.montage_images(sh)
#plt.figure()   
#plt.title('Layer 2 Convolutions') 
#plt.axis('off')
#plt.imshow(shimage)
#plt.show()

#To reload a module:
#import imp
#imp.reload(module_name)
