# -*- coding: utf-8 -*-
# @Time : 2022/4/20 16:25
# @Author : li.zhang
# @File : lstm.py


from utils.pyp import *


class RnnModel(nn.Module):
    """
    Desc: define model
    """

    def __init__(self, conf):
        """
        Desc: init class
        """
        super(RnnModel, self).__init__()
        self.input_l = conf['input_size']  # 特征的个数
        self.output_l = conf['output_size']  # 预测的长度
        self.hidden_l = conf['hidden_size']  # 隐藏层的个数
        self.layer_l = conf['layer_size']

        self.lstm = nn.GRU(input_size=self.input_l, hidden_size=self.hidden_l, num_layers=1, batch_first=True,
                           dropout=0.25)
        self.fn = nn.Linear(in_features=self.hidden_l, out_features=1)


    def forward(self, x):
        """
        Desc: forward
        Input: x[batch_size, seq_len(time_step), feature_num]
            h0[bi*num_layer, batch_size, hidden_size]
        """
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_l))
        # TODO input and output
        h_out, hn = self.lstm(x, h0)  #
        # print(h_out.size())
        out = self.fn(h_out)
        # print(out.size())

        return out