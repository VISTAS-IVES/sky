#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:51:32 2017

@author: jeffmullins
"""


import unittest
from net import BLACK, BLUE, WHITE
import net
import numpy as np


class Test_net(unittest.TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_scale(self):
        img = np.array([[BLACK, WHITE, BLUE],
                        [BLACK, BLACK, WHITE],
                        [BLUE, WHITE, WHITE]])
        correct = np.array([[[0, 0, 0], [1, 1, 1], [0, 0, 1]],
                            [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
                            [[0, 0, 1], [1, 1, 1], [1, 1, 1]]])
        img = net.scale(img)
        self.assertTrue((img == correct).all())

    def test_mask_to_one_hot(self):
        img = np.array([[BLACK, WHITE, BLUE],
                        [BLACK, BLACK, WHITE],
                        [BLUE, WHITE, WHITE]])
        one_hot = net.mask_to_one_hot(img)
        correct = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                            [[0, 0, 1], [0, 0, 1], [1, 0, 0]],
                            [[0, 1, 0], [1, 0, 0], [1, 0, 0]]])
        self.assertTrue((one_hot == correct).all())

    def test_mask_to_index(self):
        img = np.array([[BLACK, WHITE, BLUE],
                        [BLACK, BLACK, WHITE],
                        [BLUE, WHITE, WHITE]])
        indexes = net.mask_to_index(img)
        correct = np.array([[2, 0, 1],
                            [2, 2, 0],
                            [1, 0, 0]])
        self.assertTrue((indexes == correct))

    def test_one_hot_to_mask(self):
        output = np.array([[[.6, 1, 1], [2, 3, 0], [.9, 1, 0]],
                           [[0, 0, 1], [0, 0, 1], [1, 0, 0.00001]],
                           [[0, 1, 0], [1, 87, 12], [1, 3.14159265, 1]]])
        max_indexes = np.array([[1, 2, 0],
                                [0, 0, 1],
                                [1, 2, 2]])
        outs = net.one_hot_to_mask(max_indexes, output)
        correct = np.array([[BLUE, BLACK, WHITE],
                            [WHITE, WHITE, BLUE],
                            [BLUE, BLACK, BLACK]])
        self.assertTrue((outs == correct).all())


if __name__ == '__main__':
    unittest.main()
