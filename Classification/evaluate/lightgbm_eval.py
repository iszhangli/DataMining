# -*- coding: utf-8 -*-
# @Time : 2021/12/16 19:32
# @Author : li.zhang
# @File : lightgbm_eval.py


from utils.common_tools import *

import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization


import warnings
warnings.filterwarnings('ignore')



# n folds
def lgb_train_eval(train_set=None, tar_name='label', ratio=1, test_set=None, is_test=False, thd=0.5):
    """
    train_set: [feature, label]
    test_set: [feature]
    """

    X = train_set.drop(tar_name, axis=1)
    y = train_set[tar_name]

    feature_cols = [col for col in train_set.columns if col not in [tar_name]]

    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'metric': 'auc',
        'is_unbalance': False,
        'boost_from_average': False,
    }

    importance = pd.DataFrame()
    if is_test:
        prediction = np.zeros(len(test_set))
    models = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index, :], X.iloc[valid_index, :]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        weights = [ratio if val == 1 else 1 for val in y_train]
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)  # free_raw_data=True
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        model = lgb.train(params, train_data, num_boost_round=1000,
                          valid_sets=[train_data, valid_data], verbose_eval=250, early_stopping_rounds=100)

        imp_df = pd.DataFrame()
        imp_df['feature'] = feature_cols
        imp_df['split'] = model.feature_importance()
        imp_df['gain'] = model.feature_importance(importance_type='gain')
        imp_df['fold'] = fold_n + 1
        importance = pd.concat([importance, imp_df], axis=0)

        # feature_imp = list()
        # for i in range(0, importance.shape[0] // 5):
        #     m_df = list()
        #     m_df.append(importance.iloc[i, :].feature)
        #     m_df += list(importance.loc[[i]].mean().values)
        #     feature_imp.append(m_df)
        # imp_df = pd.DataFrame(feature_imp, columns=['feature', 'split', 'gain', 'fold'])
        # sort_imp_df = imp_df.sort_values(by=['gain'], ascending=False)

        models.append(model)
        if is_test == True:
            predict_y = model.predict(test_set, num_iteration=model.best_iteration)
            predict_ = model.predict(X_valid, num_iteration=model.best_iteration)
            predict_label = [1 if i > thd else 0 for i in predict_y]
            binary_classifier_metrics(y_valid, predict_label, predict_)  # every result
            prediction += predict_y
    if is_test == True:
        return models, importance, prediction
    else:
        return models, importance


# 学习曲线，一般不用
def adjust_lgb_param(train_set=None, tar_name='label', test_set=None, metric='accuracy'):

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_set.drop(tar_name, axis=1), train_set[tar_name], test_size=0.3)


    acc_train, acc_test = [], []
    pre_train, pre_test = [], []
    recall_train, recall_test = [], []
    f1_train, f1_test = [], []
    auc_train, auc_test = [], []
    map_train, map_test = [], []
    # 调优精度
    start, end, step = 1, 10, 1
    n_cv = 5

    for i in range(start, end, step):
        lgbc = LGBMClassifier(
            boosting_type='gbdt',
            class_weight='balanced',
            objective='binary',
            colsample_bytree=1.0,
            importance_type='split',  # 和feature_importance 配合使用
            n_estimators=81,
            learning_rate=0.1,
            max_depth=i,
            min_child_samples=20,
            min_child_weight=0.001,
            min_split_gain=0.0,
            num_leaves=31,
            reg_alpha=0.0,
            reg_lambda=0.0,
            silent=True,
            subsample=1.0,
            subsample_for_bin=200000,
            subsample_freq=0)
        score = cross_val_score(lgbc, Xtrain, Ytrain, cv=n_cv, scoring='accuracy').mean()
        acc_train.append(score)
        score = cross_val_score(lgbc, Xtrain, Ytrain, cv=n_cv, scoring='precision').mean()
        pre_train.append(score)
        score = cross_val_score(lgbc, Xtrain, Ytrain, cv=n_cv, scoring='recall').mean()
        recall_train.append(score)
        score = cross_val_score(lgbc, Xtrain, Ytrain, cv=n_cv, scoring='f1').mean()
        f1_train.append(score)
        score = cross_val_score(lgbc, Xtrain, Ytrain, cv=n_cv, scoring='roc_auc').mean()
        auc_train.append(score)

    print('max(accuracy):  %f, index: %d' % (max(acc_train), (acc_train.index(max(acc_train)) * step) + 1 + start))
    print('max(precision): %f, index: %d' % (max(pre_train), (pre_train.index(max(pre_train)) * step) + 1 + start))
    print('max(recall):    %f, index: %d' % (
    max(recall_train), (recall_train.index(max(recall_train)) * step) + 1 + start))
    print('max(f1):        %f, index: %d' % (max(f1_train), (f1_train.index(max(f1_train)) * step) + 1 + start))
    print('max(roc_auc):   %f, index: %d' % (max(auc_train), (auc_train.index(max(auc_train)) * step) + 1 + start))

    fig = plt.figure(figsize=[20, 4])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('accuracy')
    ax.plot(range(start, end, step), acc_train, c="red", label="accuracy")
    ax.plot(range(start, end, step), pre_train, c="green", label="precision")
    ax.plot(range(start, end, step), recall_train, c="blue", label="recall")
    ax.plot(range(start, end, step), f1_train, c="orange", label="f1")
    ax.plot(range(start, end, step), auc_train, c="pink", label="auc")
    ax.legend(fontsize="xx-large")
    plt.show()


# Adjust the param of lightGBM
# eval_result = {}
def adjust_lgb_parameters(train_set=None, tar_name='label', test_set=None):
    """[auc, binary_logloss(binary), binary_error]
    """

    epochs = 100
    x_lim = (0, 5)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_set.drop(tar_name, axis=1), train_set[tar_name],
                                                    test_size=0.3)
    train_data = lgb.Dataset(Xtrain, label=Ytrain)
    valid_data = lgb.Dataset(Xtest, label=Ytest)

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(211)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'metric': 'binary_logloss',
        'n_estimators': 100,
    }
    eval_result_ = {}
    model = lgb.train(params, train_set=train_data, verbose_eval=epochs, valid_sets=[train_data, valid_data], evals_result=eval_result_)
    lgb.plot_metric(eval_result_, metric=params['metric'], ax=ax, xlim=x_lim)

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'metric': 'binary_logloss',
        'n_estimators': 100,
        'learning_rate': 0.03,
        'num_leaves': 21,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
    }
    eval_result_last = {}
    model = lgb.train(params, train_set=train_data, verbose_eval=epochs, valid_sets=[train_data, valid_data], evals_result=eval_result_last)
    lgb.plot_metric(eval_result_last, metric=params['metric'], ax=ax, xlim=x_lim)

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'metric': 'binary_logloss',
        'n_estimators': 100,
        'learning_rate': 0.03,
        'num_leaves': 21,
        'reg_alpha': 0.2,
        'reg_lambda': 0.2,
        'subsample': 0.5
    }
    eval_result_this = {}
    model = lgb.train(params, train_set=train_data, verbose_eval=epochs, valid_sets=[train_data, valid_data], evals_result=eval_result_this)
    lgb.plot_metric(eval_result_this, metric=params['metric'], ax=ax, xlim=x_lim)
    plt.show()


def GridSearchCV_lgb(train_set=None, tar_name='label'):

    train_x, train_y = train_set.drop(tar_name, axis=1), train_set[tar_name]
    parameters = {
        # 'n_estimators': [500, 1000, 2000, 3000, 5000],
        # 'num_leaves': [10,20.30,40,50],
        # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'max_depth': [5, 10, 15, 20, 25]  # default = 31
        # 'min_child_weight': [0, 2, 5, 10, 20],
        # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
        # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
        # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
        # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    }

    lgbc = lgb.LGBMClassifier(
                            max_depth=10,
                            learning_rate=0.01,
                            n_estimators=100,
                            silent=True,
                            objective='binary',
                            min_child_weight=1,
                            max_delta_step=0,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=1440,
                            num_threads=-1,
                            verbosity=-1
                           )
    # Scoring options: [accuracy, f1, precision, recall, roc_auc]
    # https://scikit-learn.org/0.22/modules/model_evaluation.html#scoring-parameter
    gsearch = GridSearchCV(lgbc, param_grid=parameters, scoring='f1', cv=3, verbose=3, return_train_score=False)
    gsearch.fit(train_x, train_y)


    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print('Best estimator:\n', gsearch.best_estimator_)

    print('详细结果:\n', pd.DataFrame.from_dict(gsearch.cv_results_))
    print('最佳分数:\n', gsearch.best_score_)
    print('最佳参数:\n', gsearch.best_params_)


def RandomizedSearchCV_lgb(train_set=None, tar_name='label'):

    train_x, train_y = train_set.drop(tar_name, axis=1), train_set[tar_name]
    parameters = {
        # 'n_estimators': [500, 1000, 2000, 3000, 5000],
        # 'num_leaves': [10,20.30,40,50],
        # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'max_depth': [5, 10, 15, 20, 25],  # default = 31
        # 'min_child_weight': [0, 2, 5, 10, 20],
        # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
        # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
        # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
        # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    }

    lgbc = lgb.LGBMClassifier(
                            max_depth=10,
                            learning_rate=0.01,
                            n_estimators=100,
                            silent=True,
                            objective='binary',
                            min_child_weight=1,
                            max_delta_step=0,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            reg_alpha=0,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            seed=1440,
                            num_threads=-1,
                            verbosity=-1
                           )
    rand_ser = RandomizedSearchCV(lgbc, param_distributions=parameters, scoring='f1', cv=3, verbose=3)
    rand_ser.fit(train_x, train_y)

    print("Best score: %0.3f" % rand_ser.best_score_)
    print("Best parameters set:")
    best_parameters = rand_ser.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print('Best estimator_:\n', rand_ser.best_estimator_)

    print('详细结果:\n', pd.DataFrame.from_dict(rand_ser.cv_results_))
    print('最佳分数:\n', rand_ser.best_score_)
    print('最佳参数:\n', rand_ser.best_params_)


# bayes调参初探

def bayes_lgb(train_set=None, tar_name='label'):

    train_x, train_y = train_set.drop(tar_name, axis=1), train_set[tar_name]

    def lgbc_cv(n_estimators, max_depth, learning_rate,
                min_child_weight, max_delta_step, subsample,
                colsample_bytree, reg_alpha, reg_lambda, scale_pos_weight):


        val = cross_val_score(
            lgb.LGBMClassifier(max_depth=int(max_depth),
                learning_rate=min(learning_rate,0.3),
                n_estimators=int(n_estimators),
                #silent=True,
                verbose=4,
                objective='binary',
                min_child_weight=int(min_child_weight),
                max_delta_step=int(max_delta_step),
                subsample=min(subsample,0.95),
                colsample_bytree=min(colsample_bytree,0.95),
                reg_alpha=min(reg_alpha,1),
                reg_lambda=min(reg_lambda,1),
                scale_pos_weight=min(scale_pos_weight,1),
                seed=1440),
            train_x, train_y, scoring='roc_auc', cv=5
        ).mean()
        return val

    lgbc_bo = BayesianOptimization(
            lgbc_cv,
            {
                  'max_depth': (5,100),
                  'learning_rate': (0.001,0.3),
                  'n_estimators': (500,5000),
                  'min_child_weight': (0,20),
                  'max_delta_step': (0,2),
                  'subsample': (0.1,0.99),
                  'colsample_bytree': (0.1,0.99),
                  'reg_alpha': (0,1),
                  'reg_lambda': (0,1),
                  'scale_pos_weight': (0,1)

    }
        )
    lgbc_bo.maximize(n_iter=5) # 100
    print(lgbc_bo.max)


