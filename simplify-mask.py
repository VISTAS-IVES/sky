#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:23:23 2017

@author: drake
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('mouse.png')
imgplot = plt.imshow(img)
plt.show()
