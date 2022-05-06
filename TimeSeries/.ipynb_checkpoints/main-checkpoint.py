# -*- coding: utf-8 -*-
# @Time : 2022/2/10 10:37
# @Author : li.zhang
# @File : main.py


def returnNull(x, flag=''):
    if x is None:
        if flag == '1':
            return
        else:
            return
    return

r = returnNull(None, '')
print(r==None)


from arch.unitroot import ADF

adf = ADF(data)
adf.trend = 'ct'