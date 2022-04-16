# -*- coding: utf-8 -*-
# @Time : 2022/1/19 16:30
# @Author : li.zhang
# @File : smoothing.py

from utils.pyp import *


# 平稳性检验
def adf_test(data):
    res = adfuller(data)
    print(f'p value:{res[1]}')


def smoothing(data, how):
    # data is time series
    # fir:first order 一阶差分
    # second: second order  二阶差分
    # season: season order 季节差分
    if how=='fir':
        data_diff = data.diff()
    elif how == 'second':
        data_diff = data.diff().diff()

    elif how == 'senson':
        data_diff = data.diff(12).dropna()

    return data_diff


def noise_test(data):
    res = acorr_ljungbox(data, lags=[6, 12, 24], return_df=True)
    print(res)

