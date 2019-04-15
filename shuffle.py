#!/usr/bin/python
# -*- coding:utf8 -*-
import sys
if len(sys.argv) != 3:
        print('usage: python shuffle.py file_src file_tgt')
        print(sys.argv)
        exit(0)

f1 = open(sys.argv[1], 'r').readlines()
f2 = open(sys.argv[2], 'r').readlines()

o1 = open(sys.argv[1]+'.shuf', 'w')
o2 = open(sys.argv[2]+'.shuf', 'w')

import numpy as np

idx = np.random.permutation(len(f1))


for i in idx:
        o1.write(f1[i])
        o2.write(f2[i])

print('done!')
