# -*- coding: utf-8 -*-
# @Time : 2021/12/16 16:34
# @Author : li.zhang
# @File : common_tools.py


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, \
    confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import seaborn as sns
import joblib
import os
import time
import gc


# 读取数据
ds = 1
def read_data(name=None, source_path=None):
    if name is None:
        print('Source data NA. GG')
    elif name == 'dsId':
        return ds.load(source_path)
    elif name == 'df':
        return pd.read_csv(source_path, sep='\t')
    return 0


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



def reduce_memory(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                              100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df


def del_var(var_name):
    del var_name
    gc.collect()



# 二分类评估指标
# roc_auc_score = roc_curve + auc
# test_labels: true label
# predict_labels: predict lables
# predict_prob: predict output is prob
def binary_classifier_metrics(test_labels, predict_labels, predict_prob, show_flag=True):  # 评价标准
    accuracy = accuracy_score(test_labels, predict_labels)  # accuracy_score准确率
    precision = precision_score(test_labels, predict_labels)  # precision_score精确率
    recall = recall_score(test_labels, predict_labels)  # recall_score召回率
    f1_measure = f1_score(test_labels, predict_labels)  # f1_score  f1得分
    confusionMatrix = confusion_matrix(test_labels, predict_labels)  # confusion_matrix  混淆矩阵
    fpr, tpr, threshold = roc_curve(test_labels, predict_prob, pos_label=1)  # roc_curve ROC曲线
    Auc = auc(fpr, tpr)
    MAP = average_precision_score(test_labels, predict_prob)  # average_precision_score

    TP, FP, FN, TN = confusionMatrix[1, 1], confusionMatrix[0, 1], confusionMatrix[1, 0], confusionMatrix[0, 0]
    if show_flag is True:
        print("------------------------- ")
        print("row: precision | col: recall ")
        print("confusion matrix:")
        print("------------------------- ")
        print("| TP: %5d | FP: %5d | P: %5d |" % (TP, FP, TP + FP))
        print("----------------------- ")
        print("| FN: %5d | TN: %5d | R: %.3f|" % (FN, TN, (TP + FP) / len(test_labels)))
        print("----------------------- ")
        print("| T: %5d  | R: %.3f | N: %5d |" % (TP + FN, (TP + FN) / len(test_labels), len(test_labels)))
        print(" ------------------------- ")
        print("Accuracy:       %.2f%%" % (accuracy * 100))
        print("Precision:      %.2f%%" % (precision * 100))
        print("Recall:         %.2f%%" % (recall * 100))
        print("F1-measure:     %.2f%%" % (f1_measure * 100))
        print("AUC:            %.2f%%" % (Auc * 100))
        print("MAP:            %.2f%%" % (MAP * 100))
        print("------------------------- ")
    return recall, precision, f1_measure


def binary_classifier_metrics2(test_labels, predict_labels, predict_prob):  # 评价标准
    accuracy = accuracy_score(test_labels, predict_labels)  # accuracy_score准确率
    precision = precision_score(test_labels, predict_labels)  # precision_score精确率
    recall = recall_score(test_labels, predict_labels)  # recall_score召回率
    f1_measure = f1_score(test_labels, predict_labels)  # f1_score  f1得分
    auc = roc_auc_score(test_labels, predict_prob)
    return {'Accuracy': str(round(accuracy*100,2))+'%',
            'Precision:': str(round(precision*100,2))+'%',
            'Recall': str(round(recall*100,2))+'%',
            "F1-measure": str(round(f1_measure*100,2))+'%',
            "AUC": str(round(auc*100,2))+'%'}


# 分段统计函数
# y_test: ture label
# pro_y: predict probability
def segment_statistic(y_test=None, prob_y=None, bins=None):
    if bins is None:
        bins = np.arange(0, 1.1, 0.1)
    new_df = pd.DataFrame({'y_true': y_test, 'prob_y': prob_y})
    new_df['bins'] = pd.cut(new_df['prob_y'], bins)
    stra_df = new_df.groupby('bins').agg({'bins': 'count', 'y_true': 'sum'})
    stra_df.rename(columns={'bins': 'pred_cnt', 'y_true': 'real_unsat_cnt'}, inplace=True)
    stra_df = stra_df.sort_index(ascending=False)
    stra_df['pred_unsat_cnt_p'] = stra_df['pred_cnt'].cumsum()
    stra_df['recall_preson'] = stra_df['pred_unsat_cnt_p'] / stra_df['pred_cnt'].sum()
    stra_df['real_unsat_cnt_tp'] = stra_df['real_unsat_cnt'].cumsum()
    stra_df['recall'] = stra_df['real_unsat_cnt_tp'] / stra_df['real_unsat_cnt'].sum()
    stra_df['precision'] = stra_df['real_unsat_cnt_tp'] / stra_df['pred_unsat_cnt_p']
    return stra_df


# 2. the importance of feature
def show_feature_importance(model, usage_col):
    import_df = pd.DataFrame()
    import_df['feature'] = usage_col
    import_df['split'] = model.feature_importance()
    import_df['gain'] = model.feature_importance(importance_type='gain')
    import_df = import_df.sort_values(by=['gain'], ascending=False)
    return import_df


def show_feature_importance_xgb(model):
    import_df = pd.DataFrame([model.get_fscore()]).T.reset_index().sort_values([0], ascending=False)
    return import_df


def show_feature_importance_ctb(model, usage_col):
    import_df = pd.DataFrame()
    import_df['feature'] = usage_col
    import_df['importances'] = model.feature_importances_
    # import_df['gain'] = model.feature_importance(importance_type='gain')
    import_df = import_df.sort_values(by=['importances'], ascending=False)
    return import_df


def plt_feature_importance(model):
    lgb.plot_importance(model, ax=ax,
                        title='Feature Importance',
                        xlabel='Information Gain',
                        ylabel='Feature Name',
                        importance_type='gain',
                        max_num_features=10,
                        grid=False,
                        precision=3,
                        height=0.6)





# 解决余弦相似度计算值缺失的问题
# 去中心化的余弦相似度计算
# 1. 相关系数取值一般在-1~1之间
# 2. 绝对值越接近1说明变量之间的线性关系越强，绝对值越接近0说明变量间线性关系越弱。
# 3. ≥0.8高度相关，0.5~0.8中度相关，0.3~0.5低度相关，＜0.3相关关系极弱可视为不相关。
def plot_heatmap(dataframe):
    """
    """
    corr_df = dataframe
    mcorr = corr_df.corr(method="spearman")

    ax = plt.subplots(figsize=(30, 25))  # 调整画布大小
    mask = np.zeros_like(mcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # 颜色分布
    ax = sns.heatmap(mcorr, mask=mask, cmap=cmap, annot=True, fmt='.1f')  # 画热力图   annot=True


# visualization AUC
# AUC可视化
def create_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC Validation')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# 程序持久化
# joblib.dump(model, save_path)
# 加载模型
# path = os.getcwd()
# model = joblib.load()


