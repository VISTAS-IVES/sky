#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:04:42 2017

@author: drake
"""

exp = 46
i = 0
variants = ('a:conv-{0}-{1}-in b:conv-{0}-{1}-a c:conv-{0}-{1} d:maxpool-1-100-c e:maxpool-100-1-c f:concat-c-d g:concat-e-f h:concat-g-in i:conv-{0}-5-h',)

for v in variants:
    for kernel_width in (5, ):
        for channels in (16, 32, 64):
            for j in range(5):
                print('{}-{:0>2} '.format(exp, i) + v.format(kernel_width, channels))
                i += 1
