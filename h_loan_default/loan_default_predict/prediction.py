# -*- coding: utf-8 -*-
# @Time : 2021/11/17 15:36
# @Author : li.zhang
# @File : prediction.py

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score


class TrainModel(object):
    # ...

    def __init__(self):
        pass

    def train_model(self, X_train, y_train):

        train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.2)

        train_matrix = lgb.Dataset(train_x, label=train_y)
        valid_matrix = lgb.Dataset(val_x, label=val_y)
        params = {
            'boosting_type': 'gbdt',
            'class_weight': 'balanced',
            'objective': 'binary',
            'learning_rate': 0.01,
            'metric': 'auc',
            'min_child_weight': 1e-3,
            'num_leaves': 15,
            'max_depth': 12,
            'reg_lambda': 0.5,
            'reg_alpha': 0.5,
            'feature_fraction': 1,
            'bagging_fraction': 1,
            'bagging_freq': 0,
            'seed': 2020,
            'nthread': 8,
            'silent': True,
            'verbose': -1,
            'subsample': 0.5
        }

        model = lgb.train(params, train_set=train_matrix, valid_sets=valid_matrix, num_boost_round=400,
                          verbose_eval=100, early_stopping_rounds=100)

        val_pred_lgb = model.predict(val_x, num_iteration=model.best_iteration)
        fpr, tpr, threshold = metrics.roc_curve(val_y, val_pred_lgb)
        roc_auc = metrics.auc(fpr, tpr)
        import matplotlib.pyplot as plt
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.4f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()

        return model


    def adjust_lgb_parameters(self, x_train=None, y_train=None, x_test=None, y_test=None, xlim=None):
        # auc / binary_logloss(binary) / binary_error
        train_data_l = lgb.Dataset(x_train, label=y_train)
        valid_data_l = lgb.Dataset(x_test, label=y_test)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(211)
        xlim = xlim
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'metric': 'binary_logloss',
            'n_estimators': 1000,
            'max_depth': -1,
            'learning_rate': 0.015,
            'num_leaves': 21,
            'reg_alpha': 0.2,
            'reg_lambda': 0.5,
        }
        evals_result_ori = {}
        model = lgb.train(params, train_set=train_data_l, verbose_eval=100, valid_sets=[train_data_l, valid_data_l],
                          evals_result=evals_result_ori)
        lgb.plot_metric(evals_result_ori, metric=params['metric'], ax=ax, xlim=xlim)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'metric': 'binary_logloss',
            'n_estimators': 1000,
            'max_depth': -1,
            'learning_rate': 0.02,
            'num_leaves': 21,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'subsample': 0.5
        }
        evals_result_last = {}
        model = lgb.train(params, train_set=train_data_l, verbose_eval=100, valid_sets=[train_data_l, valid_data_l],
                          evals_result=evals_result_last)
        lgb.plot_metric(evals_result_last, metric=params['metric'], ax=ax, xlim=xlim)

        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'metric': 'binary_logloss',
            'n_estimators': 1000,
            'max_depth': 31,
            'learning_rate': 0.01,
            'num_leaves': 21,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'subsample': 0.5
        }
        evals_result_this = {}
        model = lgb.train(params, train_set=train_data_l, verbose_eval=100, valid_sets=[train_data_l, valid_data_l],
                          evals_result=evals_result_this)
        lgb.plot_metric(evals_result_this, metric=params['metric'], ax=ax, xlim=xlim)



