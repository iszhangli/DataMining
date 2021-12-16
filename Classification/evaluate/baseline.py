# -*- coding: utf-8 -*-
# @Time : 2021/12/16 17:21
# @Author : li.zhang
# @File : baseline.py

from utils.common_tools import binary_classifier_metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree,svm
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def baseline_model(train_set, tar_name='label'):

    X_train = train_set.drop(tar_name, axis=1)
    y_train = train_set[tar_name]
    # need to standardization -----
    X_train = X_train.fillna(0)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    # -----------------------
    train_x, val_x, train_y, val_y = train_test_split(X_train_s, y_train, test_size=0.20, random_state=42)
    # ------------------------ LogisticRegression -------
    print("----------LogisticRegression--------------- ")
    clf2 = LogisticRegression()
    clf2.fit(train_x, train_y)
    lr_y_pred = clf2.predict(val_x)
    binary_classifier_metrics(val_y, lr_y_pred, lr_y_pred)
    # ------------------------KNeighborsClassifier -------
    print("----------KNeighborsClassifier--------------- ")
    clf3 = KNeighborsClassifier(5)
    clf3.fit(train_x, train_y)
    knc_y_pred = clf3.predict(val_x)
    binary_classifier_metrics(val_y, knc_y_pred, knc_y_pred)
    # -----------------------svm.SVC ---------------------
    print("----------SVC--------- ------ ")
    clf5 = svm.SVC()
    clf5.fit(train_x, train_y)
    svm_y_pred = clf5.predict(val_x)
    binary_classifier_metrics(val_y, svm_y_pred, svm_y_pred)
    # ----------------------tree.DecisionTreeClassifier---
    print("----------DecisionTreeClassifier--------------- ")
    clf4 = tree.DecisionTreeClassifier()
    clf4 = clf4.fit(train_x, train_y)
    dtc_y_pred = clf4.predict(val_x)
    binary_classifier_metrics(val_y, dtc_y_pred, dtc_y_pred)
    # -------------------- RandomForestClassifier --------
    print("----------RandomForestClassifier--------------- ")
    clf1 = RandomForestClassifier()
    clf1.fit(train_x, train_y)
    rfc_y_pred = clf1.predict(val_x)
    binary_classifier_metrics(val_y, rfc_y_pred, rfc_y_pred)
    # -------------------- Lightgbm ----------------------
    print("----------lightgbm--------------- ")
    train_data_l = lgb.Dataset(train_x, label=train_y)
    valid_data_l = lgb.Dataset(val_x, label=val_y)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'metric': 'auc',
    }
    model = lgb.train(params, train_set=train_data_l, num_boost_round=500, verbose_eval=100, valid_sets=[train_data_l, valid_data_l], early_stopping_rounds=20)
    predict_y = model.predict(val_x) # model['cvbooster']
    predict_label = [1 if i >0.5 else 0 for i in predict_y]
    binary_classifier_metrics(val_y, predict_label, predict_y)
    # -------------------- xgboost ----------------------
    print("----------xgboost--------------- ")
    train_data_x = xgb.DMatrix(train_x, label=train_y)
    valid_data_x = xgb.DMatrix(val_x, label=val_y)
    param = {
        'objective':'binary:logistic'
    }
    bst = xgb.train(param, dtrain=train_data_x, num_boost_round=500, evals=[(valid_data_x,'eval'), (train_data_x,'train')], verbose_eval=100, early_stopping_rounds=20)
    predict_y = bst.predict(xgb.DMatrix(val_x))
    predict_label = [1 if i >0.5 else 0 for i in predict_y]
    binary_classifier_metrics(val_y, predict_label, predict_y)
    # -------------------- Catboost ----------------------
    print("----------Catboost--------------- ")
    train_pool = cb.Pool(train_x, label=train_y)
    test_pool = cb.Pool(val_x, label=val_y)
    param = {
        'objective':'Logloss'
    }
    ctb = cb.train(params=param, dtrain=train_pool, num_boost_round=500, eval_set=[test_pool, train_pool], verbose_eval=100, early_stopping_rounds=20)
    predict_y = ctb.predict(cb.Pool(val_x))
    predict_label = [1 if i >0.5 else 0 for i in predict_y]
    binary_classifier_metrics(val_y, predict_label, predict_y)