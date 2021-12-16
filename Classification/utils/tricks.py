# -*- coding: utf-8 -*-
# @Time : 2021/12/16 17:03
# @Author : li.zhang
# @File : tricks.py

# 训练集和测试集的数据是否不同
# 训练集 测试集 打标签


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import copy
import pandas  as pd
import lightgbm as lgb

# 训练集 = 1 / 测试集 = 0
# 二分类模型训练
# 如果AUC=0.5左右可以用，否则需要删除噪声
def test_dataset(origin_train=None, origin_test=None):  # TODO fix
    print(origin_train.columns)
    origin_train_df = copy.deepcopy(origin_train)
    origin_test_df = copy.deepcopy(origin_test)
    origin_train_df = origin_train_df.drop('isDefault', axis=1)
    # 打标签
    origin_train_df['u_label'] = 0
    origin_test_df['u_label'] = 1

    origin_valid = pd.concat([origin_train_df, origin_test_df], axis=0)
    origin_valid = origin_valid.drop(columns=['loan_id', 'user_id', 'class', 'employer_type', 'industry', 'work_year', 'issue_date', 'earlies_credit_mon'])
    # 简单特征处理
    # le = LabelEncoder()
    # valid_new_data['provice'] = le.fit_transform(valid_new_data.provice)
    # valid_new_data['city'] = le.fit_transform(valid_new_data.city)
    # valid_new_data['model'] = le.fit_transform(valid_new_data.model)
    # valid_new_data['make'] = le.fit_transform(valid_new_data.make)
    # 划分数据集
    train_x, val_x, train_y, val_y = train_test_split(origin_valid.iloc[:,0:-1], origin_valid.iloc[:,-1], test_size=0.3)
    # 模型训练

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc'
    }
    train_set = lgb.Dataset(train_x, train_y)
    valid_set = lgb.Dataset(val_x, val_y)
    lgb.train(params=params, train_set=train_set, valid_sets=[train_set, valid_set], num_boost_round=1000, verbose_eval=100)


# TODO 频率打分