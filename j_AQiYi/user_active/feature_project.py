import gc

import pandas as pd
import numpy as np
from common_utils import *
from itertools import groupby


input_dir = "E:/Dataset/爱奇艺用户留存预测/"
app_launch_dir = input_dir + 'app_launch_logs.csv'
user_playback_dir = input_dir + 'user_playback_data.csv'
video_info_dir = input_dir + 'video_related_data.csv'
user_info_dir = input_dir + 'user_portrait_data.csv'
user_interact_dir = input_dir + 'user_interaction_data.csv'
test_dir = input_dir + 'test_without_label.csv'

# how to check the var deleted?
app_launch = pd.read_csv(app_launch_dir)  # 194M
app_launch = reduce_memory(app_launch)  # 56.7M

test_data = pd.read_csv(test_dir)
test_data = reduce_memory(test_data)

# user app launch
launch_grp = app_launch.groupby('user_id')\
    .agg(launch_date=('date', list), launch_type=('launch_type', list))\
    .reset_index()

del app_launch
gc.collect()

# TODO add launch list

# use launch group [end_date, label] and test to contact the input data
train_data = launch_grp[["user_id", "end_date", "label"]]
test_data['label'] = -1
# This is a new df
data = pd.concat([train_data, test_data], ignore_index=True)

del train_data, test_data
gc.collect()

# use launch group [launch_type, launch_date] and test to build the feature
launch_grp = launch_grp.append(
    test_data.merge(launch_grp[["user_id", "launch_type", "launch_date"]],
                    how="left", on="user_id")
)
# TODO add launch seq








