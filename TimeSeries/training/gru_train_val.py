# -*- coding: utf-8 -*-
# @Time : 2022/4/21 14:59
# @Author : li.zhang
# @File : gru_train_val.py

# x, y -> cpu or gpu
# adjust lr
# model input and output
# class @function


from utils.pyp import *
from configs.arg_parse import parsing_args
from utils.processing import read_data
from Experience.exp_lstm import ExpLSTM


def train_and_val():
    """
    Desc:
    """
    # 1. 解析参数
    conf = parsing_args()

    # 2. 读取数据
    data_set = read_data(conf)

    # 3. 模型测试
    exp_lstm = ExpLSTM(conf, data_set)
    exp_lstm.training()

    print(conf)



if __name__ == '__main__':
    train_and_val()