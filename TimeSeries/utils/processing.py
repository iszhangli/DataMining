# -*- coding: utf-8 -*-
# @Time : 2022/1/7 19:22
# @Author : li.zhang
# @File : processing.py




from utils.pyp import *



def read_data(conf):
    """
    Desc: read data
    """
    input_path = conf['input_path_dir']
    file_name = conf['file_name']
    cols = conf['cols']
    data_raw = pd.read_csv(f'{input_path}{file_name}')
    data_raw[cols] = data_raw[cols].fillna(0)
    scaler = MinMaxScaler()
    scaler = scaler.fit(data_raw[cols])
    data_tran = scaler.transform(data_raw[cols])
    return data_tran



def average_smoothing(signal, kernel_size=3, stride=1):
    sample = []
    start = 0
    end = kernel_size

    while end < len(signal):
        start += stride
        end += stride
        me = np.mean(signal[start:end])
        re = np.ones(end-start)*me
        sample.extend(re)

    return np.array(sample)





