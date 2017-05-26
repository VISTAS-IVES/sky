#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:45:02 2017

@author: drake
"""

import unittest
from preprocess import simplify_name

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
        
if __name__ == '__main__':
    unittest.main()
