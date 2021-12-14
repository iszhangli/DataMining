# -*- coding: utf-8 -*-
# @Time : 2021/12/13 15:15
# @Author : li.zhang
# @File : 1dcnn.py

import joblib
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from copy import deepcopy as dp
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)

def norm_fit(df_1,saveM = True, sc_name = 'zsco'):  # TODO  这个归一化写的可是有点牛逼 需要借鉴学习一下
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer,QuantileTransformer,PowerTransformer
    ss_1_dic = {'zsco':StandardScaler(),
                'mima':MinMaxScaler(),
                'maxb':MaxAbsScaler(),
                'robu':RobustScaler(),
                'norm':Normalizer(),
                'quan':QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1),index = df_1.index,columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2,ss_1)

def norm_tra(df_1,ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1),index = df_1.index,columns = df_1.columns)
    return(df_2)


from torch.nn.modules.loss import _WeightedLoss
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                  pos_weight = pos_weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class TrainDataset:  # 这个地方的实现有点意思，如果可以的话，可以将任务的 df.value 都生成内置的模型
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
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


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):  #
        super(Model, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512

        cha_1_reshape = int(hidden_size/cha_1)  # 16
        cha_po_1 = int(hidden_size/cha_1/2)  # 8
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3  # 2048

        self.cha_1 = cha_1  # 256
        self.cha_2 = cha_2  # 512
        self.cha_3 = cha_3  # 512
        self.cha_1_reshape = cha_1_reshape  # 16
        self.cha_po_1 = cha_po_1  # 8
        self.cha_po_2 = cha_po_2  # 2048

        # 目的：深度神经网络在训练时使每一层的输入保持相同的分布
        # num_features – 特征维度
        self.batch_norm1 = nn.BatchNorm1d(num_features)  # 937
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))  # ( , 937) 4096  一层全连接 直接展开

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)  # 256
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x





def run_training(fold, seed):
    seed_everything(seed)

    # trn_idx = train[train['kfold'] != fold].index  # 17577
    val_idx = train[train['kfold'] == fold].index  # 4371

    train_df = train[train['kfold'] != fold].reset_index(drop=True).copy()  # (17577, 1115)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True).copy()  # (4371, 1115)

    x_train, y_train, y_train_ns = train_df[feature_cols], train_df[target_cols].values, train_df[
        target_nonsc_cols2].values  # (17577, 872)    (17577, 206)    (17577, 33)
    x_valid, y_valid, y_valid_ns = valid_df[feature_cols], valid_df[target_cols].values, valid_df[
        target_nonsc_cols2].values  # (4371, 872)   (4371, 206)   (4371, 33)
    x_test = test_[feature_cols]  # (3624, 872)  需要被预测

    # ------------ norm --------------
    col_num = list(set(feat_dic['gene'] + feat_dic['cell']) & set(feature_cols))
    col_num.sort()  # 772个特征
    x_train[col_num], ss = norm_fit(x_train[col_num], True, 'quan')  # 标准化写的不错
    x_valid[col_num] = norm_tra(x_valid[col_num], ss)
    x_test[col_num] = norm_tra(x_test[col_num], ss)

    # ------------ pca --------------
    def pca_pre(tr, va, te,
                n_comp, feat_raw, feat_new):
        pca = PCA(n_components=n_comp, random_state=42)
        tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]), columns=feat_new)
        va2 = pd.DataFrame(pca.transform(va[feat_raw]), columns=feat_new)
        te2 = pd.DataFrame(pca.transform(te[feat_raw]), columns=feat_new)
        return (tr2, va2, te2)

    pca_feat_g = [f'pca_G-{i}' for i in range(n_comp1)]
    feat_dic['pca_g'] = pca_feat_g
    x_tr_g_pca, x_va_g_pca, x_te_g_pca = pca_pre(x_train, x_valid, x_test,
                                                 n_comp1, feat_dic['gene'], pca_feat_g)
    x_train = pd.concat([x_train, x_tr_g_pca], axis=1)  # x_train (17577, 872)  x_tr_g_pca  (17577, 50)  => (17577, 922)
    x_valid = pd.concat([x_valid, x_va_g_pca], axis=1)  # x_valid (4371, 872)  x_va_g_pca  (4371, 50)
    x_test = pd.concat([x_test, x_te_g_pca], axis=1)  # x_test (3624, 872)  x_te_g_pca (3624, 50)

    pca_feat_g = [f'pca_C-{i}' for i in range(n_comp2)]
    feat_dic['pca_c'] = pca_feat_g
    x_tr_c_pca, x_va_c_pca, x_te_c_pca = pca_pre(x_train, x_valid, x_test,
                                                 n_comp2, feat_dic['cell'], pca_feat_g)
    x_train = pd.concat([x_train, x_tr_c_pca], axis=1)  # (17577, 937)
    x_valid = pd.concat([x_valid, x_va_c_pca], axis=1)  # (4371, 937)
    x_test = pd.concat([x_test, x_te_c_pca], axis=1)  # (3624, 937)

    x_train, x_valid, x_test = x_train.values, x_valid.values, x_test.values

    train_dataset = TrainDataset(x_train, y_train_ns)  # 预测266维 还是 33维  .value 还有 .df 的区别
    valid_dataset = TrainDataset(x_valid, y_valid_ns)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 128
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets_0,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5,
                                              max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_tr = nn.BCEWithLogitsLoss()  # SmoothBCEwLogits(smoothing = 0.001)
    loss_va = nn.BCEWithLogitsLoss()

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    for epoch in range(1):
        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

    model.dense3 = nn.utils.weight_norm(nn.Linear(model.cha_po_2, num_targets))
    model.to(DEVICE)

    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    loss_tr = SmoothBCEwLogits(smoothing=0.001)
    loss_va = nn.BCEWithLogitsLoss()

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    oof = np.zeros((len(train), len(target_cols)))
    best_loss = np.inf

    mod_name = f"FOLD_mod11_{seed}_{fold}_.pth"

    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_tr, trainloader, DEVICE)
        valid_loss, valid_preds = valid_fn(model, loss_va, validloader, DEVICE)
        print(f"SEED: {seed}, FOLD: {fold}, EPOCH: {epoch},train_loss: {train_loss}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:

            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), mod_name)

        elif (EARLY_STOP == True):

            early_step += 1
            if (early_step >= early_stopping_steps):
                break

    # --------------------- PREDICTION---------------------
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.load_state_dict(torch.load(mod_name))
    model.to(DEVICE)

    predictions = np.zeros((len(test_), len(target_cols)))
    predictions = inference_fn(model, testloader, DEVICE)
    return oof, predictions



def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))  # (21948, 206)
    predictions = np.zeros((len(test), len(target_cols))) # (3624, 206)

    for fold in range(NFOLDS):
        oof_, pred_ = run_training(fold, seed)  # 预测数据全  再除NFOLDS？为什么啊

        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions



train = joblib.load('train.pkl')
test_ = joblib.load('test_.pkl')
test = joblib.load('test.pkl')
feature_cols = joblib.load('feature_cols.pkl')
target_cols = joblib.load('target_cols.pkl')
target_nonsc_cols2 = joblib.load('target_nonsc_cols2.pkl')
feat_dic = joblib.load('feat_dic.pkl')



# HyperParameters
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1  # 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 1  # 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
SEED = [1, 2, 3, 4]
seed = 1
n_comp1 = 50
n_comp2 = 15

num_features = len(feature_cols) + n_comp1 + n_comp2
num_targets = len(target_cols)
num_targets_0 = len(target_nonsc_cols2)
hidden_size = 4096

oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))


tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:, i]).values())) for i in range(len(target_cols))])
tar_weight0 = np.array([np.log(i + 100) for i in tar_freq])
tar_weight0_min = dp(np.min(tar_weight0))
tar_weight = tar_weight0_min / tar_weight0
pos_weight = torch.tensor(tar_weight).to(DEVICE)


oof_, predictions_ = run_k_fold(NFOLDS, seed)  # This is main funcion.  TODO
oof += oof_ / len(SEED)
predictions += predictions_ / len(SEED)

oof_tmp = dp(oof)
oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed) + 1)
sc_dic = {}
sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:, i], oof_tmp[:, i]) for i in range(len(target_cols))])