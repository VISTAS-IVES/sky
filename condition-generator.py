#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:04:42 2017

@author: drake
"""

exp = 43
i = 0
variants = ('a:conv-{0}-{1}-in b:conv-{0}-5-a',
            'a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-5-e',
            'a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-{1}-e g:conv-{0}-5-f',
            'a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-{1}-e g:conv-{0}-{1}-f h:conv-{0}-5-g')

for v in variants:
    for kernel_width in (1, 3, 5):
        for channels in (16, 32):
            for j in range(2):
                print('{}-{:0>2} '.format(exp, i) + v.format(kernel_width, channels))
                i += 1
