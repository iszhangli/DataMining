# -*- coding: utf-8 -*-
# @Time : 2022/4/28 14:58
# @Author : li.zhang
# @File : seq2seq_train_val.py


from configs.arg_parse import parsing_args
from utils.pyp import *
from utils.processing import read_data
from Experience.exp_seq2seq import ExpSeq2Seq


def train_and_val():
    """
    Desc: ...
    """
    args = parsing_args()
    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_set = read_data(args)

    exp_seq2seq = ExpSeq2Seq(args, data_set)
    exp_seq2seq.training()



if __name__ == '__main__':
    train_and_val()