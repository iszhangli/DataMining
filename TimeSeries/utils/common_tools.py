# -*- coding: utf-8 -*-
# @Time : 2022/5/6 16:11
# @Author : li.zhang
# @File : common_tools.py

from utils.pyp import *


# 数据概述
def describe_data(train, test):
    print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
    print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')
    print('-' * 50)
    # 查看哪些列具有缺失值
    print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values.')
    print(f'The train missing column: {train.columns[train.isna().any()].tolist()}.')
    for i in train.columns[train.isna().any()].tolist():
        print(f'The missing rate of \'{i}\' is {round((train[i].isna().sum() / train.shape[0])*100, 2)}%')
    print(f'There are {test.isnull().any().sum()} columns in test dataset with missing values.')
    print(f'The test missing column: {test.columns[test.isna().any()].tolist()}.')
    for i in test.columns[test.isna().any()].tolist():
        print(f'The missing rate of \'{i}\' is {round((test[i].isna().sum() / test.shape[0])*100, 2)}%')
    # 查看数据值唯一的列
    one_value_cols = []
    one_value_cols += [col for col in train.columns if train[col].nunique() <= 1]
    one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
    print(f'There are {len(one_value_cols)} columns in train dataset with one unique value.')
    print(f'{one_value_cols} of unique values in the train set')
    print(f'There are {len(one_value_cols_test)} columns in test dataset with one unique value.')
    print(f'{one_value_cols_test} of unique values in the test set')
    print('-' * 50)
    # 查看数据缺失值情况
    nan_cols = [col for col in train.columns if train[col].isna().sum() / train.shape[0] > 0.90]
    print(f'There are {len(nan_cols)} columns in train dataset with [na value > 0.9].')
    print(f'The columns name is {nan_cols}')
    nan_clos_test = [col for col in test.columns if test[col].isna().sum() / test.shape[0] > 0.90]
    print(f'There are {len(nan_clos_test)} columns in test dataset with [na value > 0.9].')
    print(f'The columns name is {nan_clos_test}')
    print('-' * 50)
    # 列类型
    numerical_col = list(train.select_dtypes(exclude=['object']).columns)
    category_col = list(filter(lambda x: x not in numerical_col,list(train.columns)))
    print(f'The numerical columns is: {numerical_col}')
    print(f'The category columns is: {category_col}')
    return one_value_cols + nan_cols


def describe_train(train):
    print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
    print('-' * 50)
    # 查看哪些列具有缺失值
    print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values.')
    print(f'The train missing column: {train.columns[train.isna().any()].tolist()}.')
    for i in train.columns[train.isna().any()].tolist():
        print(f'The missing rate of \'{i}\' is {round((train[i].isna().sum() / train.shape[0])*100, 2)}%')

    # 查看数据值唯一的列
    one_value_cols = []
    one_value_cols += [col for col in train.columns if train[col].nunique() <= 1]

    print(f'There are {len(one_value_cols)} columns in train dataset with one unique value.')
    print(f'{one_value_cols} of unique values in the train set')
    print('-' * 50)
    # 查看数据缺失值情况
    nan_cols = [col for col in train.columns if train[col].isna().sum() / train.shape[0] > 0.90]
    print(f'There are {len(nan_cols)} columns in train dataset with [na value > 0.9].')
    print(f'The columns name is {nan_cols}')
    print('-' * 50)
    # 列类型
    numerical_col = list(train.select_dtypes(exclude=['object']).columns)
    category_col = list(filter(lambda x: x not in numerical_col,list(train.columns)))
    print(f'The numerical columns is: {numerical_col}')
    print(f'The category columns is: {category_col}')
    return one_value_cols + nan_cols
