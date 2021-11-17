# -*- coding: utf-8 -*-
# @Time : 2021/11/17 15:32
# @Author : li.zhang
# @File : description.py

import pandas as pd

class Description(object):
    # ...

    def __init__(self):
        pass

    def read_data(self, path):
        return pd.read_csv(path)

    def describe_data(self, train, test):
        print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
        print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')
        print('-' * 50)
        # 查看哪些列具有缺失值
        print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values.')
        print(f'The train missing column: {train.columns[train.isna().any()].tolist()}.')
        for i in train.columns[train.isna().any()].tolist():
            print(f'The missing rate of \'{i}\' is {train[i].isna().sum() / train.shape[0]}')
        print(f'There are {test.isnull().any().sum()} columns in test dataset with missing values.')
        print(f'The test missing column: {test.columns[test.isna().any()].tolist()}.')
        for i in test.columns[test.isna().any()].tolist():
            print(f'The missing rate of \'{i}\' is {test[i].isna().sum() / test.shape[0]}')
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
        nan_clos = [col for col in train.columns if train[col].isna().sum() / train.shape[0] > 0.90]
        print(f'There are {len(nan_clos)} columns in train dataset with [na value > 0.9].')
        nan_clos_test = [col for col in test.columns if test[col].isna().sum() / test.shape[0] > 0.90]
        print(f'There are {len(nan_clos_test)} columns in test dataset with [na value > 0.9].')
        print('-' * 50)
        numerical_col = list(train.select_dtypes(exclude=['object']).columns)
        category_col = list(filter(lambda x: x not in numerical_col, list(train.columns)))
        print(f'The numerical columns is: {numerical_col}')
        print(f'The category columns is: {category_col}')

        return one_value_cols + nan_clos

    def get_numerical_columns(self, dataframe):
        # 。。。
        return list(dataframe.select_dtypes(exclude=['object']).columns)

    def get_object_columns(self, dataframe):
        numerical_col = self.get_numerical_columns(dataframe)
        category_col = list(filter(lambda x: x not in numerical_col, list(dataframe.columns)))
        return category_col
