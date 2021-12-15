# -*- coding: utf-8 -*-
# @Time : 2021/12/14 16:33
# @Author : li.zhang
# @File : utils.py


import random
import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_curve, auc, \
    roc_auc_score, confusion_matrix, average_precision_score



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def norm_fit(df_1, saveM = True, sc_name='zsco'):
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


def binary_classifier_metrics(test_labels, predict_labels, predict_prob):  # 评价标准
    accuracy = accuracy_score(test_labels, predict_labels)  # accuracy_score准确率
    precision = precision_score(test_labels, predict_labels)  # precision_score精确率
    recall = recall_score(test_labels, predict_labels)  # recall_score召回率
    f1_measure = f1_score(test_labels, predict_labels)  # f1_score  f1得分
    auc = roc_auc_score(test_labels, predict_prob)
    return {'Accuracy': accuracy*100, 'Precision:': precision*100,
            'Recall': recall*100, "F1-measure": f1_measure*100,
            "AUC": auc*100}