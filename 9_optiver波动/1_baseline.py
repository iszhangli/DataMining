# -*- coding: utf-8 -*-
# @Time : 2021/11/3 21:54
# @Author : li.zhang
# @File : 1_baseline.py

# read data
import pandas as pd

path = 'C:/ZhangLI/Codes/DataSet/optiver-realized-volatility-prediction/'
train = pd.read_csv(path + "train.csv")
book_example = pd.read_parquet(path + 'book_train.parquet/stock_id=0')
trade_example = pd.read_parquet(path + "trade_train.parquet/stock_id=0")

