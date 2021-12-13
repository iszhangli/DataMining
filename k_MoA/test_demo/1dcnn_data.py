# -*- coding: utf-8 -*-
# @Time : 2021/12/13 14:59
# @Author : li.zhang
# @File : 1dcnn_data.py

import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


input_dir = 'C:/ZhangLI/Codes/DataSet/MoA/'


train_features = pd.read_csv(input_dir+'train_features.csv')
train_targets_scored = pd.read_csv(input_dir+'train_targets_scored.csv')
train_targets_nonscored = pd.read_csv(input_dir+'train_targets_nonscored.csv')  # 评分的二进制MoA目标
test_features = pd.read_csv(input_dir+'test_features.csv')
sample_submission = pd.read_csv(input_dir+'sample_submission.csv')
train_drug = pd.read_csv(input_dir+'train_drug.csv')


# 来评分  评个锤子的分
target_cols = train_targets_scored.drop('sig_id', axis=1).columns.values.tolist()
target_nonsc_cols = train_targets_nonscored.drop('sig_id', axis=1).columns.values.tolist()
nonctr_id = train_features.loc[train_features['cp_type']!='ctl_vehicle','sig_id'].tolist()
tmp_con1 = [i in nonctr_id for i in train_targets_scored['sig_id']]
#tmp_con1


sc_dic = {}
feat_dic = {}
mat_cor = pd.DataFrame(np.corrcoef(train_targets_scored.drop('sig_id',axis = 1)[tmp_con1].T,
                      train_targets_nonscored.drop('sig_id',axis = 1)[tmp_con1].T))
mat_cor2 = mat_cor.iloc[(train_targets_scored.shape[1]-1):,0:train_targets_scored.shape[1]-1]
mat_cor2.index = target_nonsc_cols
mat_cor2.columns = target_cols
mat_cor2 = mat_cor2.dropna()
mat_cor2_max = mat_cor2.abs().max(axis = 1)

q_n_cut = 0.9
target_nonsc_cols2 = mat_cor2_max[mat_cor2_max > np.quantile(mat_cor2_max,q_n_cut)].index.tolist()
print(len(target_nonsc_cols2))

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]
feat_dic['gene'] = GENES
feat_dic['cell'] = CELLS

# sample norm
q2 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
train_features[feat_dic['gene']] = (train_features[feat_dic['gene']].T - qmean.values).T
q2 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['gene']].apply(np.quantile,axis = 1,q = 0.75).copy()
qmean = (q2+q7)/2
test_features[feat_dic['gene']] = (test_features[feat_dic['gene']].T - qmean.values).T

q2 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = train_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T - qmean.values).T
qmean2 = train_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
train_features[feat_dic['cell']] = (train_features[feat_dic['cell']].T / qmean2.values).T.copy()

q2 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.25).copy()
q7 = test_features[feat_dic['cell']].apply(np.quantile,axis = 1,q = 0.72).copy()
qmean = (q2+q7)/2
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T - qmean.values).T
qmean2 = test_features[feat_dic['cell']].abs().apply(np.quantile,axis = 1,q = 0.75).copy()+4
test_features[feat_dic['cell']] = (test_features[feat_dic['cell']].T / qmean2.values).T.copy()

# remove ctl
train = train_features.merge(train_targets_scored, on='sig_id')
train = train.merge(train_targets_nonscored[['sig_id']+target_nonsc_cols2], on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
target = train[['sig_id']+target_cols]
target_ns = train[['sig_id']+target_nonsc_cols2]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()



train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
target = train[['sig_id']+target_cols]
target_ns = train[['sig_id']+target_nonsc_cols2]

train0 = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# ---
# drug ids
tar_sig = target['sig_id'].tolist()
train_drug = train_drug.loc[[i in tar_sig for i in train_drug['sig_id']]]
target = target.merge(train_drug, on='sig_id', how='left')

# LOCATE DRUGS
vc = train_drug.drug_id.value_counts()
vc1 = vc.loc[vc <= 19].index
vc2 = vc.loc[vc > 19].index

feature_cols = []
for key_i in feat_dic.keys():
    value_i = feat_dic[key_i]
    print(key_i,len(value_i))
    feature_cols += value_i
len(feature_cols)
from copy import deepcopy as dp
feature_cols0 = dp(feature_cols)

oof = np.zeros((len(train), len(target_cols)))
predictions = np.zeros((len(test), len(target_cols)))


import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

SEED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
def seed_everything(seed=42):
    random.seed(seed)  # seed 确定 随机数确定
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU设定种子
    torch.cuda.manual_seed(seed)  # GPU设定种子
    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
    # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    torch.backends.cudnn.deterministic = True
# for seed in SEED:
#     print(seed)
seed = 1
seed_everything(1)

# sig_id	cp_time	cp_dose	g-0	g-1	g-2	g-3	g-4	g-5	g-6	...	niemann-pick_c1-like_1_protein_antagonist	omega_3_fatty_acid_stimulant	quorum_sensing_signaling_modulator	reducing_agent	ror_inverse_agonist	sars_coronavirus_3c-like_protease_inhibitor	selective_estrogen_receptor_modulator_(serm)	sphingosine_1_phosphate_receptor_agonist	steryl_sulfatase_inhibitor	tyrosine_phosphatase_inhibitor
#  0	id_000644bb2	24	D1	1.051913	0.547612	-0.257988	-0.630888	-0.204488	-1.022088	-1.032088	...	0
folds = train0.copy()
feature_cols = dp(feature_cols0)
# sig_id	5-alpha_reductase_inhibitor	11-beta-hsd1_inhibitor	acat_inhibitor	acetylcholine_receptor_agonist	acetylcholine_receptor_antagonist	acetylcholinesterase_inhibitor	adenosine_receptor_agonist	adenosine_receptor_antagonist	adenylyl_cyclase_activator	...	tyrosine_kinase_inhibitor	ubiquitin_specific_protease_inhibitor	vegfr_inhibitor	vitamin_b	vitamin_d_receptor_agonist	wnt_inhibitor	drug_id_x	drug_id_y	drug_id_x	drug_id_y
#  0	id_000644bb2	0	0	0
target2 = target.copy()

dct1 = {}; dct2 = {}
skf = MultilabelStratifiedKFold(n_splits = 5)

tmp = target2.groupby('drug_id')[target_cols].mean().loc[vc1]
tmp_idx = tmp.index.tolist()
tmp_idx.sort()
tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
tmp = tmp.loc[tmp_idx2]
for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
    dd = {k:fold for k in tmp.index[idxV].values}
    dct1.update(dd)

# STRATIFY DRUGS MORE THAN 19X
skf = MultilabelStratifiedKFold(n_splits = 5) # , shuffle = True, random_state = seed
tmp = target2.loc[target2.drug_id.isin(vc2)].reset_index(drop = True)
tmp_idx = tmp.index.tolist()
tmp_idx.sort()
tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))
tmp = tmp.loc[tmp_idx2]
for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[target_cols])):
    dd = {k:fold for k in tmp.sig_id[idxV].values}
    dct2.update(dd)

target2['kfold'] = target2.drug_id.map(dct1)
target2.loc[target2.kfold.isna(),'kfold'] = target2.loc[target2.kfold.isna(),'sig_id'].map(dct2)
target2.kfold = target2.kfold.astype(int)

folds['kfold'] = target2['kfold'].copy()

train = folds.copy()
test_ = test.copy()


# HyperParameters
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

n_comp1 = 50
n_comp2 = 15

num_features=len(feature_cols) + n_comp1 + n_comp2
num_targets=len(target_cols)
num_targets_0=len(target_nonsc_cols2)
hidden_size=4096

def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)


tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
tar_weight0_min = dp(np.min(tar_weight0))
tar_weight = tar_weight0_min/tar_weight0
pos_weight = torch.tensor(tar_weight).to(DEVICE)

# train
# test_
# test
# feature_cols
# target_cols
# target_nonsc_cols2

import joblib
joblib.dump(train, 'train.pkl')
joblib.dump(test_, 'test_.pkl')
joblib.dump(test, 'test.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')
joblib.dump(target_cols, 'target_cols.pkl')
joblib.dump(target_nonsc_cols2, 'target_nonsc_cols2.pkl')
joblib.dump(feat_dic, 'feat_dic.pkl')



# def run_k_fold(NFOLDS, seed):
#     oof = np.zeros((len(train), len(target_cols)))
#     predictions = np.zeros((len(test), len(target_cols)))
#
#     for fold in range(NFOLDS):
#         oof_, pred_ = run_training(fold, seed)
#
#         predictions += pred_ / NFOLDS
#         oof += oof_
#
#     return oof, predictions
#
#
# oof_, predictions_ = run_k_fold(NFOLDS, seed)  # This is main funcion.
# oof += oof_ / len(SEED)
# predictions += predictions_ / len(SEED)
#
# oof_tmp = dp(oof)
# oof_tmp = oof_tmp * len(SEED) / (SEED.index(seed) + 1)
# sc_dic[seed] = np.mean([log_loss(train[target_cols].iloc[:, i], oof_tmp[:, i]) for i in range(len(target_cols))])
