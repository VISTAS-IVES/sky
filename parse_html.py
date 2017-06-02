#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:33:14 2017

@author: drake
"""
from bs4 import BeautifulSoup
from datetime import datetime

with open("data/c1_20160414_20160415.html") as f:
    html = ''.join(f.readlines())
soup = BeautifulSoup(html, 'html5lib')

for elem in soup(text='OBSERVATION TIME:'):
    date = elem.parent.parent.td.text
    stamp = datetime.strptime(date, '%m/%d/%Y %H:%M GMT').strftime('%Y%m%d%H%M')
    print (stamp)