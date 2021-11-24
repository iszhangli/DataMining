# -*- coding: utf-8 -*-
# @Time : 2021/9/14 20:24
# @Author : li.zhang
# @File : 1_baseline.py

import pandas as pd
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import time
import pandas as pd
import warnings
from collections import defaultdict
import collections
import math

warnings.filterwarnings('ignore')

path = 'E:/Dataset/新闻推荐比赛数据/'
train = path + 'articles.csv'
train_click = path + 'train_click_log.csv'
# 使用历史浏览 点击文章的数据信息预测用户未来的点击行为，即用户最后一次点击新闻的文章
# 问题转换 30万用户 200w点击 36w篇文章
train_df = pd.read_csv(train)
train_click_df = pd.read_csv(train_click)
# 全量训练集
# all_click_df = get_all_click_df(data_path, offline=False)

import copy

all_click_df = copy.deepcopy(train_click_df)


def get_user_item_time(all_click_df):
    c = all_click_df.sort_values('click_timestamp')

    # 这应该是个键值对list(zip(c['click_article_id'], c['click_timestamp']))
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = c.groupby('user_id')['click_article_id', 'click_timestamp'].apply(
        lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict


user_item_time_dict = get_user_item_time(all_click_df)
# 计算物品相似度
i2i_sim = {}
item_cnt = defaultdict(int)
ii = 0
for user, item_time_list in user_item_time_dict.items():
    # print(user, item_time_list)
    # print('-' * 10)
    # 基于商品的协同过滤
    for i, i_click_time in item_time_list:
        # print(i, i_click_time)
        item_cnt[i] += 1
        i2i_sim.setdefault(i, {})  # 当key不存在时，设置为默认值，key存在时，返回值
        for j, j_click_time in item_time_list:
            if (i == j):
                continue
            i2i_sim[i].setdefault(j, 0)
            i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)  # 1/log(n+1)
# {30760: {157507: 0.01076245713904683, 211442: 0.05262861609293018}}
i2i_sim_ = i2i_sim.copy()
for i, related_items in i2i_sim.items():
    print(i, related_items)
    for j, wij in related_items.items():
        i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])  # 30760:311   两篇文章成对出现的值 / ，每篇文章出现的次数

