# -*- coding: utf-8 -*-
# @Time : 2021/11/17 15:43
# @Author : li.zhang
# @File : loan_main.py

import pandas as pd
import numpy as np
import copy
from a_8_loan_default.loan_default_predict.description import *
from a_8_loan_default.loan_default_predict.prediction import *
from a_8_loan_default.loan_default_predict.preprocessor import *
from a_8_loan_default.loan_default_predict.print_metric import *


def main():
    # ...
    preprocessor = Preprocessor()
    train_model = TrainModel()

    # ---------- common ------------
    # read data
    data_dir = 'C:/ZhangLI/Codes/DataSet/个人违贷/official_data/'
    # data_dir = 'E:/Dataset/个人违贷/official_data/'
    train_pub = pd.read_csv(data_dir + 'train_public.csv')
    train_net = pd.read_csv(data_dir + 'train_internet.csv')
    test_pub = pd.read_csv(data_dir + 'test_public.csv')

    train_net['isDefault'] = train_net['is_default']
    common_feature = list(set(train_pub.columns).intersection(set(train_net.columns)))
    # train_pub_new = copy.deepcopy(train_pub)
    # train_net_new = copy.deepcopy(train_net)
    train_pub_new = train_pub[common_feature]
    train_net_new = train_net[common_feature]
    common_feature.remove('isDefault')
    test_pub_new = test_pub[common_feature]
    # ------------------------------

    # Test 1  Only use train public data

    # Test 2  Only use train internet data
    train_net_new = preprocessor.label_encode(train_net_new)
    drop_cols = ['loan_id', 'user_id', 'policy_code',  'scoring_low', 'scoring_high', 'f1', 'early_return_amount_3mon', 'earlies_credit_mon']
    train_net_new = preprocessor.drop_cols(train_net_new, drop_cols)
    X_train = train_net_new.drop('isDefault', axis=1)
    y_train = train_net_new['isDefault']
    lgb_model = train_model.train_model(X_train, y_train)


    # test 3  Use train public and train internet
    # test 4  Up for you


if __name__ == '__main__':
    main()