# -*- coding: utf-8 -*-
# @Time : 2021/12/16 15:16
# @Author : li.zhang
# @File : main.py

import pandas as pd

from evaluate.DCNN_eval import d_cnn_train_eval
from evaluate.DNN_eval import dnn_train_eval
from evaluate.baseline import baseline_model
from evaluate.lightgbm_eval import *


def main():
    print('Read data.')
    data_dir = 'C:/ZhangLI/Codes/DataSet/public-data/classification/'
    origin_train = pd.read_csv(data_dir + 'train.csv')
    origin_test = pd.read_csv(data_dir + 'test.csv')

    print('train-evaluate baseline')
    # baseline_model(origin_train, 'isDefault')

    print('train-evaluate LightGBM')
    # lgb_train_eval(origin_train, 'isDefault')

    print('adjustment lightGBM parameter')
    # adjust_lgb_parameters(origin_train, 'isDefault')
    # GridSearchCV_lgb(origin_train, 'isDefault')
    # RandomizedSearchCV_lgb(origin_train, 'isDefault')
    bayes_lgb(origin_train, 'isDefault')


    print('train-evaluate 1D-CNN')
    # d_cnn_train_eval(origin_train, origin_test, 'isDefault')

    print('train-evaluate DNN')
    # dnn_train_eval(origin_train, origin_test, 'isDefault')









if __name__ == '__main__':
    """This is main function.
    """
    main()




