#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:54:34 2017

@author: drake
"""

import os
from preprocess import extract_timestamp

for file in os.listdir('data/cldmask/'):
    t = extract_timestamp(file)
    if 