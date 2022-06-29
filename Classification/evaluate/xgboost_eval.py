# -*- coding: utf-8 -*-
# @Time : 2021/12/16 19:46
# @Author : li.zhang
# @File : xgboost_eval.py

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def adjustment_xgb_parameters(x_train=None, y_train=None, fig_ylim=3):
    """x_train, y_train"""
    fig,ax = plt.subplots(1,figsize=(15,5))
    ax.set_ylim(top=fig_ylim)
    ax.grid()

    dfull = xgb.DMatrix(x_train,y_train)
    # Init parameter
    param1 = {'verbosity':1, # -- global parameter
              'objective':'binary:logistic',  # -- task parameter
              'eval_metric':'auc',
              "subsample":1,  # -- tree booster parameter
              "max_depth":6,
              "eta":0.3,
              "gamma":0,
              "lambda":1,
              "alpha":0,
              "colsample_bytree":1,
              "colsample_bylevel":1,
              "colsample_bynode":1,
            }
    num_round = 200
    cvresult1 = xgb.cv(params=param1, dtrain=dfull, num_boost_round=num_round,nfold=5)
    ax.plot(range(1,num_round+1),cvresult1.iloc[:,0],c="red",label="train,original")
    ax.plot(range(1,num_round+1),cvresult1.iloc[:,2],c="orange",label="test,original")

    # Usable parameter
    param2 = {'verbosity':1,
              'objective':'binary:logistic',
              'eval_metric':'auc'
             }
    num_round = 200
    cvresult2 = xgb.cv(params=param1, dtrain=dfull, num_boost_round=num_round,nfold=5)
    ax.plot(range(1,num_round+1),cvresult2.iloc[:,0],c="green",label="train,last")
    ax.plot(range(1,num_round+1),cvresult2.iloc[:,2],c="blue",label="test,last")

    # Adjusting parameter
    param3 = {'verbosity':1,
              'objective':'binary:logistic',
              'eval_metric':'auc'
             }
    num_round = 200
    cvresult3 = xgb.cv(params=param1, dtrain=dfull, num_boost_round=num_round,nfold=5)
    ax.plot(range(1,num_round+1),cvresult3.iloc[:,0],c="gray",label="train,this")
    ax.plot(range(1,num_round+1),cvresult3.iloc[:,2],c="pink",label="test,this")
    ax.legend(fontsize="xx-large")
    plt.show()
# train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
# adjustment_xgb_parameters(train_x, train_y)

def adjust_lgb_parameters(train_set=None, tar_name='label', test_set=None):
    """[auc, binary_logloss(binary), binary_error]
    """

    epochs = 100
    n_estimators = 1000
    x_lim = (0, n_estimators+10)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(train_set.drop(tar_name, axis=1), train_set[tar_name],
                                                    test_size=0.3)
    train_data = xgb.DMatrix(Xtrain,Ytrain)
    valid_data = xgb.DMatrix(Xtest,Ytest)

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    params = {
        'objective': 'binary:logistic',  # -- task parameter
        'eval_metric': 'logloss',  # auc
        "subsample": 1,  # -- tree booster parameter
        "max_depth": 6,
        "eta": 0.3,
        "gamma": 0,
        "lambda": 1,
        "alpha": 0,
        "colsample_bytree": 1,
        "colsample_bylevel": 1,
        "colsample_bynode": 1,
    }
    eval_result_ = {}
    model = xgb.train(params, dtrain=train_data, verbose_eval=epochs,
                      evals=[(train_data, 'train'), (valid_data, 'test')], evals_result=eval_result_,
                      num_boost_round=n_estimators)
    ax.plot(range(1, n_estimators + 1), eval_result_['train']['logloss'], c="red", label="train,original")
    ax.plot(range(1, n_estimators + 1), eval_result_['test']['logloss'], c="orange", label="test,original")

    params = {'verbosity':1,
              'objective':'binary:logistic',
              'eval_metric':'logloss'
             }
    eval_result_last = {}
    model = xgb.train(params, dtrain=train_data, verbose_eval=epochs,
                      evals=[(train_data, 'train'), (valid_data, 'test')], evals_result=eval_result_last,
                      num_boost_round=n_estimators)
    ax.plot(range(1, n_estimators + 1), eval_result_last['train']['logloss'], c="green", label="train,last")
    ax.plot(range(1, n_estimators + 1), eval_result_last['test']['logloss'], c="blue", label="test,last")

    params = {'verbosity':1,
              'objective':'binary:logistic',
              'eval_metric':'logloss'
             }
    eval_result_this = {}
    model = xgb.train(params, dtrain=train_data, verbose_eval=epochs,
                      evals=[(train_data, 'train'), (valid_data, 'test')], evals_result=eval_result_this,
                      num_boost_round=n_estimators)
    ax.plot(range(1, n_estimators + 1), eval_result_this['train']['logloss'], c="gray", label="train,last")
    ax.plot(range(1, n_estimators + 1), eval_result_this['test']['logloss'], c="pink", label="test,last")

    ax.legend(fontsize="xx-large")
    plt.show()