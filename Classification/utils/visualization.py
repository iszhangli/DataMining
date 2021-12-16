# -*- coding: utf-8 -*-
# @Time : 2021/12/16 15:25
# @Author : li.zhang
# @File : visualization.py


# 测试集和训练集连续值分布
import matplotlib.pyplot as plt
import seaborn as sns
def plot_feature_kde(train_data, test_data, features=[]):
    """连续变量"""
    plt.clf()
    fcols = 4
    frows = len(features)
    plt.figure(figsize=(6*fcols, frows*1.5))
    i = 0
    for col in features:
        i += 1
        plt.subplot(frows//fcols+1, fcols, i)
        sns.kdeplot(train_data[col], color="Blue", shade=True, label='Train')
        sns.kdeplot(test_data[col], color="Red", shade=True, label='Test')
    plt.show()


# 正负样本占比
def show_pie():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    colors = ['#ff9999', 'lightgrey']  # 自定义颜色
    plt.pie(x=[491475, 47757205],
            explode=[0.1, 0.07],
            labels=['5G潜客', '非5G潜客'],
            colors=colors,
            autopct='%.2f%%')
    plt.title('正负样本占比')
