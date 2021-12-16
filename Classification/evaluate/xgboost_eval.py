# -*- coding: utf-8 -*-
# @Time : 2021/12/16 19:46
# @Author : li.zhang
# @File : xgboost_eval.py


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
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
adjustment_xgb_parameters(train_x, train_y)