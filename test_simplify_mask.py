#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:51:18 2017

@author: drake
"""

import unittest
from simplify_mask import BLACK, BLUE, GREEN, YELLOW, WHITE, GRAY
import simplify_mask
import numpy as np

class TestSimplifyMask(unittest.TestCase):
    
    def setup(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_list_colors(self):
        img = np.array([[[0, 1, 2], [3, 4, 5]],
                        [[3, 4, 5], [6, 7, 8]]])
        colors = simplify_mask.list_colors(img)
        correct = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertTrue((colors == correct).all())
    
    def test_simplify_colors(self):
        # Colors from mask images
        """
        BLACK = np.array([0, 0, 0])
        BLUE = np.array([0, 0, 255])
        GREEN = np.array([0, 255, 0])
        GRAY = np.array([192, 192, 192])
        YELLOW = np.array([255, 255, 0])
        WHITE = np.array([255, 255, 255])
        """
        img = np.array([[BLACK, BLUE, GREEN],
                        [GRAY, YELLOW, WHITE]])
        colors = simplify_mask.simplify_colors(img)
        correct = np.array([[BLACK, BLUE, BLACK],
                          [GRAY, BLACK, WHITE]])
        self.assertTrue((colors == correct).all())

if __name__ == '__main__':
    unittest.main()
