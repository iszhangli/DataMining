# -*- coding: utf-8 -*-
# @Time : 2022/4/20 15:57
# @Author : li.zhang
# @File : arg_parse.py

import yaml

def parsing_args():
    """
    Desc: [] -> {}
    """
    with open("../configs/const.yml", encoding='utf8') as yaml_file:

        args = yaml.load(yaml_file, Loader=yaml.FullLoader)
        # print(args)

    return args


if __name__ == '__main__':

    parsing_args()

