# -*- coding: utf-8 -*-
# @Time : 2021/12/14 16:33
# @Author : li.zhang
# @File : train.py


from test_nn.utils import *
from test_nn.dataset import TestDataset, TrainDataset
from test_nn.model import Model, SmoothBCEwLogits

from sklearn.model_selection import StratifiedKFold
import pandas as pd

import torch
from torch import nn
import torch.optim as optim



def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        targets = torch.reshape(targets, [targets.shape[0], 1])
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()  # TODO how to use it?

        final_loss += loss.item()

    final_loss /= len(dataloader)  # The loss sum / mean / change dataloader to data

    return final_loss

def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        targets = torch.reshape(targets, [targets.shape[0], 1])
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds


def run_train(train_dataset, valid_dataset, seed=2021):   # or name run_train
    seed_everything(seed)
    # read data

    # HyperParameters
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 20
    EARLY_STOPPING_STEPS = 10
    EARLY_STOP = False

    num_features = 37
    num_targets_0 = 1
    hidden_size = 512

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 128
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets_0,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # set the learning rate of Each param group
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5,
                                              max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_tr = SmoothBCEwLogits(smoothing=0.001)
    # loss_tr = nn.BCEWithLogitsLoss()
    loss_va = nn.BCEWithLogitsLoss()  # the mean of logistic

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    best_loss = np.inf

    mod_name = f"FOLD_mod11_{seed}_{seed}_.pth"

    for epoch in range(EPOCHS):
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)  # completed data TODO
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)  # The model parameter TODO
        print(f"SEED: {seed}, FOLD: {seed}, EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:

            best_loss = valid_loss
            # oof[val_idx] = valid_preds
            torch.save(model.state_dict(), mod_name)  # TODO  deepcopy(mode.state_dict()) ?

        elif (EARLY_STOP == True):

            early_step += 1
            if (early_step >= early_stopping_steps):
                break


def main():

    print(f'Read origin data.')
    data_dir = 'C:/ZhangLI/Codes/DataSet/个人违贷/'
    # data_dir = 'E:/Dataset/个人违贷/'
    train_ = pd.read_csv(data_dir + 'default_train.csv')
    test_ = pd.read_csv(data_dir + 'default_test.csv')

    x_train = train_.drop(['isDefault'], axis=1)
    y_train = train_['isDefault']
    feature_cols = x_train.columns.to_list()
    target_cols = ['isDefault']
    # print(len(feature_cols))

    # normalization
    print(f'Normalization.')
    x_train[feature_cols], ss = norm_fit(x_train[feature_cols], True, 'mima')  # another
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