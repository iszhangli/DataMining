# 模型程序或者模型的说明
1. 地址：https://www.kaggle.com/c/m5-forecasting-accuracy/overview

## 数据说明
1. sales_train_evaluation.csv  
```
item_id: The id of the product.
dept_id: The id of the department the product belongs to.
cat_id: The id of the category the product belongs to.
store_id: The id of the store where the product is sold.
state_id: The State where the store is located.
d_1, d_2, …, d_i, … d_1941: The number of units sold at day i, starting from 2011-01-29. 
```
2. sales_train_validation.csv
2. sample_submission.csv
3. sell_prices.csv
```
store_id: The id of the store where the product is sold.
item_id: The id of the product.
wm_yr_wk: The id of the week.
sell_price: The price of the product for the given week/store. The price is provided per week (average across seven days). If not available, this means that the product was not sold during the examined week. Note that although prices are constant at weekly basis, they may change through time (both training and test set). 
```
4. calendar.csv
```
ate: The date in a “y-m-d” format.
wm_yr_wk: The id of the week the date belongs to.
weekday: The type of the day (Saturday, Sunday, …, Friday).
wday: The id of the weekday, starting from Saturday.
month: The month of the date.
year: The year of the date.
event_name_1: If the date includes an event, the name of this event.
event_type_1: If the date includes an event, the type of this event.
event_name_2: If the date includes a second event, the name of this event.
event_type_2: If the date includes a second event, the type of this event.
snap_CA, snap_TX, and snap_WI: A binary variable (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP3 purchases on the examined date. 1 indicates that SNAP purchases are allowed. 
```


## 思路
1. 开始可以是行形式，再应用到数据集时封装为函数

### 特征可视化


### 时间平稳序列的检验及方法
为什么要求是平稳序列？
平稳序列应该如何理解呢？
1. 时序图/可视化检验[]
2. 分段统计均值和方法
跟新移动平均函数？？？
3. 可视化统计特征 ACF-自相关系数 PACF-偏自相关系数
4. 假设检验的方法 单位根 DF-test ADF-test PP-test DF-gls kpss

### 白噪声检验

### 小波变化


### 非平稳序列转为平稳序列  
函数封装 ？？？？
1. 差分 一阶差分 二阶差分
2. 平滑
3. 变换
4. 分解

### 算法
1. naive approach
2. Weighted Moving Average
3. Exponential Smoothing
3.1 Single Exponential Smoothing
3.2 Holt
3.2.1 Holt线性趋势模型
3.2.2 Holt乘法趋势模型
3.3 Holt-winter
3.3.1 Holt-Winters加法模型
3.3.2 Holt-Winters乘法模型
3.3.3 Holt-Winters的衰减法 能预测多长时间的序列
5. ARMA
6. ARIMA
7. SARIMA
8. LGBM
9. RF
10. CATBOOST
11. XGB
12. LSTM

8. Prophet

n-beats 
seq2seq
wavenet 
 



