# -*- coding: utf-8 -*-
# @Time : 2022/8/12 11:45
# @Author : li.zhang
# @File : test_main.py

from sklearn.metrics import log_loss

y_true = ["spam", "ham", "ham", "spam"]
y_pred = [[.1, .9], [.9, .1], [.8, .2], [.35, .65]]

n = log_loss(y_true, y_pred)

print(n)


n = log_loss(["ham", "spam", "spam", "ham"], [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])

print(n)

import math
print(math.log10(0.9) + math.log10(0.1) + math.log10(0.2) + math.log10(0.65))
