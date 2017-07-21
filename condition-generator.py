#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:04:42 2017

@author: drake
"""

exp = 45
i = 0
variants = ('a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-5-e',
            'a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-{1}-e g:concat-a-f h:conv-{0}-5-g',
            'a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-{1}-e g:conv-{0}-{1}-f h:concat-a-g i:conv-{0}-5-h',
            'a:conv-{0}-{1}-in b:maxpool-1-100-a c:maxpool-100-1-a d:concat-a-b e:concat-c-d f:conv-{0}-{1}-e g:conv-{0}-{1}-f h:conv-{0}-{1}-g i:concat-a-h j:conv-{0}-5-i')

for v in variants:
    for kernel_width in (5, ):
        for channels in (16, 32):
            for j in range(2):
                print('{}-{:0>2} '.format(exp, i) + v.format(kernel_width, channels))
                i += 1
