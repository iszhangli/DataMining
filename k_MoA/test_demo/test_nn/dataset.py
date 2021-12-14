# -*- coding: utf-8 -*-
# @Time : 2021/12/14 16:33
# @Author : li.zhang
# @File : dataset.py

import torch

class TrainDataset:  # 这个地方的实现有点意思，如果可以的话，可以将任务的 df.value 都生成内置的模型
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx], dtype=torch.float)
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct