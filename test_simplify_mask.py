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
        img = np.array([[BLACK, BLUE, GREEN],
                        [GRAY, YELLOW, WHITE]])
        colors = simplify_mask.simplify_colors(img)
        correct = np.array([[BLACK, BLUE, BLACK],
                          [WHITE, BLACK, WHITE]])
        self.assertTrue((colors == correct).all())

    def test_color_counts(self):
        img = np.array([[BLACK, BLUE, BLACK],
                        [WHITE, BLACK, WHITE]])
        probs = simplify_mask.color_counts(img)
        correct = np.array([1, 2, 3])
        self.assertTrue((probs == correct).all())
    
    def test_simplify_name_skyimage(self):
        f = 'sgptsiskyimageC1.a1.20160414.235930.jpg.20160414235930.jpg'
        self.assertEqual(simplify_mask.simplify_name(f), 'skyimage20160414235930.jpg')
        
    def test_simplify_name_cldmask(self):
        f = 'sgptsicldmaskC1.a1.20160414.235930.png.20160414235930.png'
        self.assertEqual(simplify_mask.simplify_name(f), 'cldmask20160414235930.png')
        
    def test_separate_data(self):
        data = list(range(100))
        numbers = simplify_mask.separate_data(data)
        # This will fail in the rare event that shuffling does nothing
        self.assertNotEqual(numbers, list(range(100)))
        numbers.sort() # For equality testing below
        self.assertEqual(numbers, list(range(100)))
        
if __name__ == '__main__':
    unittest.main()
