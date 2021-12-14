# -*- coding: utf-8 -*-
# @Time : 2021/12/14 16:33
# @Author : li.zhang
# @File : train.py


from test_nn.utils import *
from test_nn.dataset import TestDataset, TrainDataset
from test_nn.model import Model

from sklearn.model_selection import StratifiedKFold
import pandas as pd

import torch
from torch import nn

def train_fn():
    pass


def run_train(train_dataset, valid_dataset, seed=2021):   # or name run_train
    seed_everything(seed)
    # read data

    # HyperParameters
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 128
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets_0,
        hidden_size=hidden_size,
    )



def main():

    print(f'Read origin data.')
    data_dir = 'C:/ZhangLI/Codes/DataSet/个人违贷/'
    train_ = pd.read_csv(data_dir + 'default_train.csv', index_col='Unnamed: 0')
    test_ = pd.read_csv(data_dir + 'default_test.csv', index_col='Unnamed: 0')

    x_train = train_.drop(['isDefault'], axis=1)
    y_train = train_['isDefault']
    feature_cols = x_train.columns.to_list()
    target_cols = ['isDefault']

    # normalization
    print(f'Normalization.')
    x_train[feature_cols], ss = norm_fit(x_train[feature_cols], True, 'quan')  # 标准化写的不错
    test_[feature_cols] = norm_tra(test_[feature_cols], ss)

    SEED = [1]
    NFOLDS = 2
    folds = StratifiedKFold(NFOLDS)

    for seed in SEED:
        print(f'Seed : {seed}.')
        for train_index, valid_index in folds.split(x_train, y_train):
            t_x_train, t_y_train = x_train.iloc[train_index, :], y_train.iloc[train_index]
            t_x_valid, t_y_valid = x_train.iloc[valid_index, :], y_train.iloc[valid_index]

            t_x_train, t_y_train = t_x_train.values, t_y_train.values
            t_x_valid, t_y_valid = t_x_valid.values, t_y_valid.values

            train_dataset = TrainDataset(t_x_train, t_y_train)  # 二分类 ， 多分类？
            valid_dataset = TrainDataset(t_x_valid, t_y_valid)

            run_train(train_dataset, valid_dataset, seed)

if __name__ == '__main__':

    main()