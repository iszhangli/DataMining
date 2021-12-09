
import pandas as pd
import numpy as np
from itertools import groupby
import gc



def get_end_date(list_date):
    n1, n2 = min(list_date), max(list_date)
    if n1 < n2 - 7:
        end_date = np.random.randint(n1, n2 - 7)  # 110 140  seq between end_date-30 end_date
    else:
        end_date = np.random.randint(100, 222 - 7)   # data_min
    return end_date


def get_label(row):
    # TODO
    # the other method to build the label / about test data
    launch_list = row.launch_date
    end = row.end_date
    label = sum([1 for x in set(launch_list) if end < x < end + 8])
    return label


def gen_launch_seq(row):
    seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
    seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
    end = row.end_date
    seq = [seq_map.get(x, 0) for x in range(end - 31, end + 1)]
    return seq


def target_encoding(name, df, m=1):
    # target encoding
    df[name] = df[name].astype('string').str.split(";")
    df = df.explode(name)
    overall = df["label"].mean()
    df = df.groupby(name).agg(
        freq=("label", "count"),
        in_category=("label", np.mean)
    ).reset_index()
    df["weight"] = df["freq"] / (df["freq"] + m)
    df["score"] = df["weight"] * df["in_category"] + (1 - df["weight"]) * overall
    return df

def get_playtime_seq(row):
    seq_sort = sorted(zip(row.playtime_list, row.date_list), key=lambda x: x[1])
    seq_map = {k: sum(x[0] for x in g) for k, g in groupby(seq_sort, key=lambda x: x[1])}
    seq_norm = {k: 1/(1+np.exp(3-v/450)) for k, v in seq_map.items()}
    seq = [round(seq_norm.get(i, 0), 4) for i in range(row.end_date-31, row.end_date+1)]
    return seq

def get_duration_prefer(duration_list):
    drn_list = sorted(duration_list.split(";"))
    drn_map = {k: sum(1 for _ in g) for k, g in groupby(drn_list) if k != "nan"}
    if drn_map:
        max_ = max(drn_map.values())
        res = [round(drn_map.get(str(i)+'.0', 0)/max_, 4) for i in range(1, 17)]   # TODO
        return res
    else:
        return np.nan

def get_id_score(id_list):
    # TODO check the score f
    global id_score
    x = sorted(id_list.split(";"))
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x) if k != "nan"}
    if x_count:
        x_sort = sorted(x_count.items(), key=lambda k: -k[1])
        top_x = x_sort[:3]
        res = [(n, id_score.get(k, 0)) for k, n in top_x]  # 需要特别注意 get函数是否能得到正确的值
        res = sum(n*v for n, v in res) / sum(n for n, v in res)
        return res
    else:
        return np.nan

def target_id_score(column1, column2, column3, df):
    id_score = dict()
    father_df = df.loc[(df.label >= 0) & (df[column1].notna()), [column1, "label"]]
    father_id_score = target_encoding(column1, father_df)
    id_score.update({x[1]: x[5] for x in father_id_score.itertuples()})
    del father_df, father_id_score
    gc.collect()

    tag_df = df.loc[(df.label >= 0) & (df[column2].notna()), [column2, "label"]]
    tag_id_score = target_encoding(column2, tag_df)
    id_score.update({x[1]: x[5] for x in tag_id_score.itertuples()})
    del tag_df, tag_id_score
    gc.collect()

    cast_df = df.loc[(df.label >= 0) & (df[column3].notna()), [column3, "label"]]
    cast_id_score = target_encoding(column3, cast_df)
    id_score.update({x[1]: x[5] for x in cast_id_score.itertuples()})
    del cast_df, cast_id_score
    gc.collect()

    return id_score


def get_interact_prefer(interact_type):
    x = sorted(interact_type)
    x_count = {k: sum(1 for _ in g) for k, g in groupby(x)}
    x_max = max(x_count.values())
    res = [round(x_count.get(i, 0)/x_max, 4) for i in range(1, 12)]
    return res



