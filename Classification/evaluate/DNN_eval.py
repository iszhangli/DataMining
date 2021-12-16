# -*- coding: utf-8 -*-
# @Time : 2021/12/16 15:23
# @Author : li.zhang
# @File : DNN_eval.py

from utils.preprocess import *
from utils.nn_tools import *
from utils.common_tools import *
from models.DNN import DNN

from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from torch import nn
import numpy as np
import copy


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
        scheduler.step()  #

        final_loss += loss.item()

    final_loss /= len(dataloader)  # The loss sum / mean / change dataloader to data

    return final_loss


def valid_fn(model, loss_fn, dataloader, device, thd=0.5):
    model.eval()
    final_loss = 0
    valid_preds = []
    labels = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        targets = torch.reshape(targets, [targets.shape[0], 1])
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy().reshape(-1))
        labels.append(targets.detach().cpu().numpy().reshape(-1))

    final_loss /= len(dataloader)


    labels = np.concatenate(labels)
    valid_preds = np.concatenate(valid_preds)
    preds_label = [1 if i>thd else 0 for i in valid_preds]
    result = binary_classifier_metrics2(labels, preds_label, valid_preds)

    return final_loss, valid_preds, result



def dnn_train_eval(train=None, test=None, tar_name='label'):
    """1d-cnn
    """
    train_ = train  # copy.deepcopy()
    test_ = test

    # Set parameters
    SEED = [1]
    NFOLDS = 2
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    EPOCHS = 20

    early_stopping_steps = 10
    early_step = 0
    EARLY_STOP = False


    x_train = train_.drop(tar_name, axis=1)
    y_train = train_[tar_name]

    feature_cols = x_train.columns

    print(f'Normalization.')
    x_train[feature_cols], ss = norm_fit(x_train[feature_cols], True, 'quan')  # another
    test_[feature_cols] = norm_tra(test_[feature_cols], ss)

    for seed in SEED:
        folds = StratifiedKFold(NFOLDS)
        fold = 0
        for train_index, valid_index in folds.split(x_train, y_train):
            fold += 1
            t_x_train, t_y_train = x_train.iloc[train_index, :], y_train.iloc[train_index]
            t_x_valid, t_y_valid = x_train.iloc[valid_index, :], y_train.iloc[valid_index]

            t_x_train, t_y_train = t_x_train.values, t_y_train.values
            t_x_valid, t_y_valid = t_x_valid.values, t_y_valid.values

            train_dataset = TrainDataset(t_x_train, t_y_train)  # 二分类 ， 多分类？
            valid_dataset = TrainDataset(t_x_valid, t_y_valid)

            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

            model = DNN(
                num_features=len(feature_cols),
                num_targets=1,
            )

            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5,
                                                      max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(trainloader))

            loss_tr = SmoothBCEwLogits(smoothing=0.001)
            loss_va = nn.BCEWithLogitsLoss()  # the mean of logistic

            best_loss = np.inf
            mod_name = f"1d-cnn_{seed}_{fold}.pth"

            for epoch in range(EPOCHS):
                train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)  # completed data
                valid_loss, valid_preds, result = valid_fn(model, loss_va, validloader,
                                                           DEVICE)  # The model parameter TODO
                print(f"SEED: {seed}, FOLD: {fold}, EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")
                print(result)

                if valid_loss < best_loss:

                    best_loss = valid_loss
                    torch.save(copy.deepcopy(model.state_dict()), mod_name)

                elif (EARLY_STOP == True):

                    early_step += 1
                    if (early_step >= early_stopping_steps):
                        break

    return model


