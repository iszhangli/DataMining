
import pandas as pd
import numpy as np
from itertools import groupby


class DataPreProcess(object):
    # 为了删除中间结果，暂不定义 属性
    def __init__(self):
        pass

    @staticmethod
    def get_end_date(self, list_date):
        n1, n2 = min(list_date), max(list_date)
        if n1 < n2 - 7:
            end_date = np.random.randint(n1, n2 - 7)
        else:
            end_date = np.random.randint(100, 222 - 7)
        return end_date

    @staticmethod
    def get_label(self, row):
        # TODO
        # the other method to build the label / about test data
        launch_list = row.launch_date
        end = row.end_date
        label = sum([1 for x in set(launch_list) if end < x < end + 8])
        return label

    @staticmethod
    def gen_launch_seq(self, row):
        seq_sort = sorted(zip(row.launch_type, row.launch_date), key=lambda x: x[1])
        seq_map = {k: max(g)[0] + 1 for k, g in groupby(seq_sort, lambda x: x[1])}
        end = row.end_date
        seq = [seq_map.get(x, 0) for x in range(end - 31, end + 1)]
        return seq



