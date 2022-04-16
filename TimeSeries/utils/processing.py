# -*- coding: utf-8 -*-
# @Time : 2022/1/7 19:22
# @Author : li.zhang
# @File : processing.py




from utils.pyp import *

def average_smoothing(signal, kernel_size=3, stride=1):
    sample = []
    start = 0
    end = kernel_size

    while end < len(signal):
        start += stride
        end += stride
        me = np.mean(signal[start:end])
        re = np.ones(end-start)*me
        sample.extend(re)

    return np.array(sample)


single = np.array([2,3,4,6,7,3,8,6,4,7,4,3,8,9,6])
print(average_smoothing(single))

import lightgbm as lgb

