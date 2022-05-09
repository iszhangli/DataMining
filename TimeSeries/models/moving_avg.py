# -*- coding: utf-8 -*-
# @Time : 2022/1/19 16:09
# @Author : li.zhang
# @File : moving_avg.py


from utils.pyp import *

# Simple Moving Average
# Weighted Moving Average
# Exponential Moving Average




def weighted_moving_average(data, k=1, w=3, method='sma'):
    """
    Desc: 预测k个，窗口大小为w
        'sma': 移动平均；
        'wma': 加权移动平均；
        'ema'：指数移动平均
    """
    res = []
    if method == 'sma':
        for i in range(k):
            num = np.mean(list(data[-w+i:])+res[:i])
            res.append(num)
    elif method == 'wma':
        seq = np.array(range(1, w+1))
        sum_seq = np.sum(seq)
        weights = seq/sum_seq
        print(weights)
        for i in range(k):
            num = [a*b for a,b in zip(list(data[-w+i:])+res[:i], weights)]
            res.append(np.sum(num))
    elif method == 'ema':
        seq = np.array(range(1, 10+1))
        seqs = np.power(seq,2)
        sum_seq = np.sum(seqs)
        weights = seq/sum_seq
        print(weights)
        for i in range(k):
            num = [a*b for a,b in zip(list(data[-w+i:])+res[:i], weights)]
            res.append(np.sum(num))
    return res

