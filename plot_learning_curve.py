#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:30:05 2017

@author: drake
"""

import matplotlib.pyplot as plt
import sys

with open(sys.argv[1]) as f:
    data = f.readlines()

data = data[1:-1] # Strip off first and last line
x = [row.split('\t')[0] for row in data]
y = [row.split('\t')[1] for row in data]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Minibatches')
ax1.set_ylabel('Accuracy')
ax1.plot(x,y)
plt.show()
