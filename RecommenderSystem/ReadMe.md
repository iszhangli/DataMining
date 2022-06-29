### Thinking: 
1. 构建合适的评价指标

### Goal: 减少马太效应，需要在历史上很少接触的产品上表现良好

### 字段信息
1. 文件组织
 |--underexpose_train.zip
   |--underexpose_item_feat.csv 
     |--item_id
     |--txt_vec
     |--img_vec
   |--underexpose_user_feat.csv
     |--user_id
     |--user_age_level
     |--user_gender
     |--user_city_level

underexpose_test_click-T.csv：user_id, item_id, time
underexpose_test_qtime-T.csv：user_id, query_time`需要预测qtime的点击时间`
T=0,1,2,…,6用于开发，参与者的最终排名将根据 T=7,8,9 计算。

2. 字段信息
item_id：商品的唯一标识符
txt_vec：项目的文本特征，它是由预先训练的模型生成的128维实值向量
img_vec：项目的图像特征，它是由预先训练的模型生成的128维实值向量
user_id：用户的唯一标识符
time：   点击事件发生的时间戳，即（unix_timestamp-random_number_1）/ random_number_2
user_age_level： 用户所属的年龄段
user_gender：    用户的性别，可以为空
user_city_level：用户所在城市的等级

3. 解压信息
7c2d2b8a636cbd790ff12a007907b2ba underexpose_train_click-1
ea0ec486b76ae41ed836a8059726aa85 underexpose_train_click-2
65255c3677a40bf4d341b0c739ad6dff underexpose_train_click-3
c8376f1c4ed07b901f7fe5c60362ad7b underexpose_train_click-4
63b326dc07d39c9afc65ed81002ff2ab underexpose_train_click-5
f611f3e477b458b718223248fd0d1b55 underexpose_train_click-6
ec191ea68e0acc367da067133869dd60 underexpose_train_click-7
90129a980cb0a4ba3879fb9a4b177cd2 underexpose_train_click-8
f4ff091ab62d849ba1e6ea6f7c4fb717 underexpose_train_click-9

96d071a532e801423be614e9e8414992 underexpose_test_click-1
503bf7a5882d3fac5ca9884d9010078c underexpose_test_click-2
dd3de82d0b3a7fe9c55e0b260027f50f underexpose_test_click-3
04e966e4f6c7b48f1272a53d8f9ade5d underexpose_test_click-4
13a14563bf5528121b8aaccfa7a0dd73 underexpose_test_click-5
dee22d5e4a7b1e3c409ea0719aa0a715 underexpose_test_click-6
69416eedf810b56f8a01439e2061e26d underexpose_test_click-7
55588c1cddab2fa5c63abe5c4bf020e5 underexpose_test_click-8
caacb2c58d01757f018d6b9fee0c8095 underexpose_test_click-9

4. submit
|--underexpose_submit-T.csv
    |--user_id, item_id_01, item_id_02, ..., item_50


### 评价指标
NDCG50-full
NDCG50-


### 工程脉络

|--recall_main.py
    |--[read_data]->[top_200_sim_item]->[top_50_click]