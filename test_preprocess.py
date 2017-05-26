#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:45:02 2017

@author: drake
"""

import unittest
from preprocess import simplify_name, extract_timestamp, simplify_colors, color_counts, separate_stamps
from preprocess import BLACK, BLUE, GREEN, YELLOW, WHITE, GRAY
import numpy as np

class TestPreprocess(unittest.TestCase):
    
    def setup(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_simplify_name_skyimage(self):
        f = 'sgptsiskyimageC1.a1.20160414.235930.jpg.20160414235930.jpg'
        self.assertEqual(simplify_name(f), 'skyimage20160414235930.jpg')
        
    def test_simplify_name_cldmask(self):
        f = 'sgptsicldmaskC1.a1.20160414.235930.png.20160414235930.png'
        self.assertEqual(simplify_name(f), 'cldmask20160414235930.png')
    
    def test_extract_timestamp(self):
        f = 'sgptsicldmaskC1.a1.20160414.235930.png.20160414235930.png'
        self.assertEqual(extract_timestamp(f), '20160414235930')
        
    def test_simplify_colors(self):
        img = np.array([[BLACK, BLUE, GREEN],
                        [GRAY, YELLOW, WHITE]])
        colors = simplify_colors(img)
        correct = np.array([[BLACK, BLUE, BLACK],
                          [WHITE, BLACK, WHITE]])
        self.assertTrue((colors == correct).all())
        
    def test_color_counts(self):
        img = np.array([[BLACK, BLUE, BLACK],
                        [WHITE, BLACK, WHITE]])
        probs = color_counts(img)
        correct = np.array([1, 2, 3])
        self.assertTrue((probs == correct).all())
        
    def test_separate_stamps(self):
        data = list(range(100))
        test, valid, train = separate_stamps(data)
        self.assertEqual(len(test), 20)
        self.assertEqual(len(valid), 16)
        self.assertEqual(len(train), 64)
        numbers = test + valid + train
        # This will fail in the rare event that shuffling does nothing
        self.assertNotEqual(numbers, list(range(100)))
        numbers.sort() # For equality testing below
        self.assertEqual(numbers, list(range(100)))
        
if __name__ == '__main__':
    unittest.main()
