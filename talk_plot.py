#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:45:01 2017

@author: drake
"""

import matplotlib.pyplot as plt
import sys
import numpy as np

data = np.empty((25, 20))

# Read in data
for i in range(25):
    with open("results/exp35-" + str(i).rjust(2, '0') + "/output.txt") as f:
        lines = f.readlines()
    lines = lines[1:]  # Strip off header line
    data[i, :] = [row.split('\t')[2] for row in lines]

print(data)

means = np.empty((5, 20))
for i in range(5):
    means[i, :] = np.mean(data[5*i:5*i+5], axis=0)

print(means)

plt.xkcd()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Minibatches of 50 training images')
ax1.set_ylabel('Accuracy')
h = list(range(5))
x = range(100, 2001, 100)
colors = 'bgrcm'
conditions = ('12 weights', '84 weights', '227 weights', '1763 weights', '96707 weights')
for i in range(5):
    h[i], = ax1.plot(x, means[i], colors[i], label=conditions[i])
#ax1.legend(handles=h, loc=4)

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
lgd = ax1.legend(handles=h, loc='center left', bbox_to_anchor=(1, 0.5))

#plt.show()
plt.tight_layout()
#fig.set_size_inches(8, 5)
fig.savefig('results/talk_plot.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')

