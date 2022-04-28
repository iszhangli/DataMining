# -*- coding: utf-8 -*-
# @Time : 2022/4/20 15:52
# @Author : li.zhang
# @File : gru_dataset.py

from utils.pyp import *


class GruDataset(Dataset):

    """
        Desc: Data
            ...
        """

    def __init__(self, conf, data, flag):
        """
        Desc:
        Input: DataSet
        """
        self.data_set = data
        self.train_size = conf['train_size']
        self.val_size = conf['val_size']
        self.test_size = conf['test_size']
        self._start = [0, 153 * 144 - 144, 169 * 144 - 144]  # [0, 21888, 24192]
        self._end = [153 * 144, 169 * 144, 184 * 144]  # [22032, 24336, 26496]

        self.type_map = {'train': 0, 'val': 1, 'test': 2}

        clip_start = 0
        clip_end = 0
        if flag == 'train':
            clip_start = self._start[self.type_map[flag]]
            clip_end = self._end[self.type_map[flag]]
        elif flag == 'val':
            clip_start = self._start[self.type_map[flag]]
            clip_end = self._end[self.type_map[flag]]

        self._data_set = self.data_set[clip_start:clip_end]


    def __getitem__(self, index):
        """
        Desc:

        """
        s_begin = index
        s_end = index + 300  # 使用200个点预测288个点
        e_begin = s_end
        e_end = s_end + 288
        seq_x = self._data_set[s_begin:s_end]
        seq_y = self._data_set[e_begin:e_end]
        return seq_x, seq_y

    def __len__(self):
        """
        Desc:

        """
        # return int((len(self._data_set)-200)/288)
        return int(len(self._data_set) - 300 - 288 + 1)  # 数据的长度
