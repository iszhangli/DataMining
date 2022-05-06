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

# 可视化
from itertools import cycle

# adf test
from statsmodels.tsa.stattools import adfuller
# 白噪声
from statsmodels.stats.diagnostic import acorr_ljungbox


## model
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F

import random
import os

##