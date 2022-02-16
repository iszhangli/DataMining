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
1. 根据数据进行操作