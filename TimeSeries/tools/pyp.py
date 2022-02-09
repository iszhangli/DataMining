# -*- coding: utf-8 -*-
# @Time : 2022/1/5 16:56
# @Author : li.zhang
# @File : pyp.py


import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize

from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

# adf test
from statsmodels.tsa.stattools import adfuller
# 白噪声
from statsmodels.stats.diagnostic import acorr_ljungbox