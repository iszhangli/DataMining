import gc

import pandas as pd
import numpy as np
from common_utils import *
from itertools import groupby
from feature_utils import *


def app_launch_processing(app_launch_data):
    app_launch = app_launch_data
    # user app launch
    launch_grp = app_launch.groupby('user_id') \
        .agg(launch_date=('date', list), launch_type=('launch_type', list)) \
        .reset_index()

    del app_launch
    gc.collect()

    # add [end_date, label]
    launch_grp['end_date'] = launch_grp.launch_date.apply(get_end_date)
    launch_grp["label"] = launch_grp.apply(get_label, axis=1)

    return launch_grp


def user_playback_processing(user_playback_data):
    pass


def main():

    # input_dir = "C:/ZhangLI/Codes/DataSet/爱奇艺用户留存预测/"
    # input_dir = "E:/Dataset/爱奇艺用户留存预测/"
    input_dir = '/home/zzs/aqy_user_retention/data/'
    app_launch_dir = input_dir + 'app_launch_logs.csv'
    user_playback_dir = input_dir + 'user_playback_data.csv'
    video_info_dir = input_dir + 'video_related_data.csv'
    user_info_dir = input_dir + 'user_portrait_data.csv'
    user_interact_dir = input_dir + 'user_interaction_data.csv'
    test_dir = input_dir + 'test_without_label.csv'

    # how to check the var deleted?
    print('Read the app launch data.')
    app_launch = pd.read_csv(app_launch_dir)  # 194M
    app_launch = reduce_memory(app_launch)  # 56.7M

    launch_grp = app_launch_processing(app_launch)

    print('Read the test data.')
    test_data = pd.read_csv(test_dir)
    test_data = reduce_memory(test_data)

    # use launch group [end_date, label] and test contact as the input data
    print('Data columns: [user_id, end_date, label]')
    train_data = launch_grp[["user_id", "end_date", "label"]]
    test_data['label'] = -1
    # This is a new df
    data = pd.concat([train_data, test_data], ignore_index=True)  # ["user_id", "end_date", "label"]

    # use launch group [launch_type, launch_date] and test to build the feature
    test_data = test_data.merge(launch_grp[["user_id", "launch_type", "launch_date"]],
                                how="left", on="user_id")
    launch_grp = launch_grp.append(test_data)
    # launch_seq
    launch_grp["launch_seq"] = launch_grp.apply(gen_launch_seq, axis=1)  # TODO [launch_type, launch_date]没用了吗？
    print(f'launch group columns {launch_grp.columns}')

    del train_data, test_data
    gc.collect()


    data = data.merge(
        launch_grp[["user_id", "end_date", "label", "launch_seq"]],
        on=["user_id", "end_date", "label"],
        how="left"
    )

    # read user playback
    user_playback = pd.read_csv(user_playback_dir)  # 2168M
    user_playback = reduce_memory(user_playback)  #

    user_playback = user_playback.merge(data, how="inner", on="user_id")

    print('Drop the user playback not between end_data-31 and end_date')
    playback = user_playback.loc[(user_playback.date >= user_playback.end_date - 31)
                                 & (user_playback.date <= user_playback.end_date)]
    del user_playback, launch_grp
    gc.collect()

    video_related = pd.read_csv(video_info_dir)  #
    video_related = reduce_memory(video_related)  #
    user_playback = playback.merge(video_related[video_related.item_id.notna()], how="left", on="item_id")

    print(f'The columns of playback between [end_date-31, end_date] {user_playback.columns}')

    print('Target encode columns are [father_id, tag_list, cast]')
    id_score = target_id_score('father_id', 'tag_list', 'cast', user_playback)  #

    playback_grp = user_playback.groupby(["user_id", "end_date", "label"]).agg(
        playtime_list=("playtime", list),
        date_list=("date", list),
        duration_list=("duration", lambda x: ";".join(map(str, x))),
        father_id_list=("father_id", lambda x: ";".join(map(str, x))),
        tag_id_list=("tag_list", lambda x: ";".join(map(str, x))),
        cast_list=("cast", lambda x: ";".join(map(str, x)))
    ).reset_index()

    del user_playback
    gc.collect()

    playback_grp["playtime_seq"] = playback_grp.apply(get_playtime_seq, axis=1)
    playback_grp["duration_prefer"] = playback_grp.duration_list.apply(get_duration_prefer)

    playback_grp["father_id_score"] = playback_grp.father_id_list.apply(get_id_score)
    playback_grp["cast_id_score"] = playback_grp.cast_list.apply(get_id_score)
    playback_grp["tag_score"] = playback_grp.tag_list.apply(get_id_score)

    print(playback_grp.columns)
    data = data.merge(
        playback_grp[
            ["user_id", "end_date", "label", "playtime_seq", "duration_prefer", "father_id_score", "cast_id_score",
             "tag_score"]],
        on=["user_id", "end_date", "label"],
        how="left"
    )
    del playback_grp
    gc.collect()

    print('Read the user device info')
    portrait = pd.read_csv(user_info_dir, dtype={"territory_code": str})
    portrait = reduce_memory(portrait)
    portrait = pd.merge(data[["user_id", "label"]], portrait, how="left", on="user_id")
    print(f'launch group columns {portrait.columns}')

    df = portrait.loc[(portrait.label >= 0) & (portrait.territory_code.notna()), ["territory_code", "label"]]  # territory code -> area code
    territory_score = target_encoding("territory_code", df)
    id_score.update({x[1]: x[5] for x in territory_score.itertuples()})

    del df, territory_score
    gc.collect()

    portrait["territory_score"] = portrait.territory_code.apply(
        lambda x: id_score.get(x, 0) if isinstance(x, str) else np.nan)

    portrait["device_ram"] = portrait.device_ram.apply(
        lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)
    portrait["device_rom"] = portrait.device_rom.apply(
        lambda x: float(x.split(";")[0]) if isinstance(x, str) else np.nan)

    data = data.merge(portrait.drop("territory_code", axis=1), how="left", on=["user_id", "label"])
    del portrait
    gc.collect()

    print('Read the user interact.') # [user_id, item_id, interact_type, date]
    user_interact = pd.read_csv(user_interact_dir, dtype={"territory_code": str})
    user_interact = reduce_memory(user_interact)

    interact_grp = user_interact.groupby("user_id").agg(
        interact_type=("interact_type", list)
    ).reset_index()
    interact_grp["interact_prefer"] = interact_grp.interact_type.apply(get_interact_prefer)  # be love

    data = data.merge(interact_grp[["user_id", "interact_prefer"]], on="user_id", how="left")

    print('Feature normalization.')
    norm_cols = ["father_id_score", "cast_id_score", "tag_score",
                 "device_type", "device_ram", "device_rom", "sex",
                 "age", "education", "occupation_status", "territory_score"]
    for col in norm_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std

    data.fillna({
        "playtime_seq": str([0] * 32),
        "duration_prefer": str([0] * 16),
        "interact_prefer": str([0] * 11)
    }, inplace=True)

    data.fillna(0, inplace=True)

    print('Save file.')
    data.loc[data.label >= 0].to_csv("./train_data.txt", sep="\t", index=False)
    data.loc[data.label < 0].to_csv("./test_data.txt", sep="\t", index=False)

if __name__ == '__main__':
    main()







