#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:51:18 2017

@author: drake
"""

import unittest
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

if __name__ == '__main__':
    unittest.main()