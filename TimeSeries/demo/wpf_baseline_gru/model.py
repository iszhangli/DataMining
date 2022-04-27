# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A GRU-based baseline model to forecast future wind power
Authors: Lu,Xinjiang (luxinjiang@baidu.com), Li,Yan (liyan77@baidu.com)
Date:    2022/03/10
"""
import paddle
import paddle.nn as nn


class BaselineGruModel(nn.Layer):
    """
    Desc:
        A simple GRU model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=True)
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc: [32, 144, 10] ->
        Returns:
            A tensor
        """
        x = paddle.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]])  # [32, 288, 10]
        x_enc = paddle.concat((x_enc, x), 1)  # [32, 432, 10]
        x_enc = paddle.transpose(x_enc, perm=(1, 0, 2))  # [432, 32, 10]
        dec, _ = self.lstm(x_enc) # [432, 32, 48(hidden_size)]  和 batch_size和 seq_len(step_len)没有关系
        dec = paddle.transpose(dec, perm=(1, 0, 2))  # [32, 432, 48]
        sample = self.projection(self.dropout(dec))   # [32, 432, 1]
        sample = sample[:, -self.output_len:, -self.out:]  # [32, 288, 1]
        return sample  # [B, L, D]
