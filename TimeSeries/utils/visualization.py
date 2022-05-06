# -*- coding: utf-8 -*-
# @Time : 2022/5/5 15:34
# @Author : li.zhang
# @File : visualization.py


from utils.pyp import *
from configs.arg_parse import parsing_args
from utils.processing import read_data


pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# sales_train_val.set_index('id')[d_cols].T.reset_index().rename(columns={'index':'d'}).merge(calendar, how='left', validate='1:1').set_index('date')
# sales_train_val_new['v'][1000:1200].plot(figsize=(15, 5), color=next(color_cycle), title='s')

if __name__ == '__main__':
    args = parsing_args()

    data_set = read_data(args)

    data_set['Patv'].plot(figsize=(15, 5), color=next(color_cycle), title='Patv')