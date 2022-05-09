# -*- coding: utf-8 -*-
# @Time : 2022/1/19 16:08
# @Author : li.zhang
# @File : naive_approach.py


def naive_approach(data, k=1):
    """
    Desc: 获取近k个点作为预测点
    """
    return data[-k:]