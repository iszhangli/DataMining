j ### 第10名方案 
* [https://zhuanlan.zhihu.com/p/149061129]
* [http://xtf615.com/2020/06/17/KDD-CUP-2020-Debiasing-Rush/]

### Thinking: 
1. 一个典型的序列推荐场景
当前推荐的内容是不是在当前的phase中出现过
构建合适的评价指标

### Goal: 减少马太效应，需要在历史上很少接触的产品上表现良好

### 字段信息
1. 文件组织
 |--underexpose_train.zip
   |--underexpose_item_feat.csv 
     |--item_id：        商品的唯一标识符
     |--txt_vec：        项目的文本特征，它是由预先训练的模型生成的128维实值向量
     |--img_vec：        项目的图像特征，它是由预先训练的模型生成的128维实值向量
   |--underexpose_user_feat.csv
     |--user_id：        用户的唯一标识符
     |--user_age_level： 用户所属的年龄段
     |--user_gender：    用户的性别，可以为空
     |--user_city_level：用户所在城市的等级
   |--train_click-0.csv
     |--user_id：        用户的唯一标识符
     |--time：           点击事件发生的时间戳，即（unix_timestamp-random_number_1）/ random_number_2
     |--item_id：        商品的唯一标识符

2. 文件组织
|--articles.csv：新闻文章信息数据表
  |--article_id           ：文章id，与click_article_id相对应
  |--category_id          ：文章类型id
  |--created_at_ts        ：文章创建时间戳
  |--words_count          ：文章字数
|--train_click_log.csv：训练集用户点击日志
  |--user_id              ：用户id
  |--click_article_id     ：点击文章id
  |--click_timestamp      ：点击时间戳
  |--click_environment    ：点击环境
  |--click_deviceGroup    ：点击设备组
  |--click_os             ：点击操作系统
  |--click_country        ：点击城市
  |--click_region         ：点击地区
  |--click_referrer_type  ：点击来源类型
|--articles_emb.csv：新闻文章embedding向量表示
  |--emb_1,emb_2,…,emb_249：文章embedding向量表示

1. 基本信息+embedding信息

underexpose_test_click-T.csv：user_id, item_id, time
underexpose_test_qtime-T.csv：user_id, query_time`需要预测qtime的点击时间`
T=0,1,2,…,6用于开发，参与者的最终排名将根据 T=7,8,9 计算。

### 特征构成
1. embedding只用1维
2. 用部分数据

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


## 工程思路
### 1、EDA思路
* 有多少用户 `df.user_id.nunique()`
* 有多少篇文章 `df.artical_id.nunique()`
* 用户点击文章的次数分布 `df.groupby(['user_id'])['click_article_id'].count().plot()` or `plt.plot(df.groupby(['user_id'])['click_article_id'].count().sort_values().values)` 根据点击次数和点击时间定义用户的活跃度
* 文章被点击的次数 `df.groupby('click_article_id')['user_id'].count()`  根据点击次数和时间进行文章热度的划分
* 文章共频次数（文章A出现后B出现的次数）
```
tmp = click_log_df.sort_values('click_timestamp')
tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x:x.shift(-1))
tmp.groupby(['click_article_id','next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)
```
* 文章信息【引申出其他特征的统计】
* 不同类型的文章出现的次数 `df.groupby('category_id')['user_id'].agg({'count'}).sort_values('count', ascending=False).reset_index().plot()`
* 用户点击文章的偏好：查看用户的兴趣是否广泛，*不能单纯的看，和用户点击文章的数量有关系* `df.groupby('user_id')['category_id'].nunique().plot()`
* 用户查看文章的长度 `df.groupby('user_id')['words_count'].agg({'mean'}).sort_values('mean', ascending=False).reset_index()['mean'].plot()` 反应用户对长文感兴趣还是长文感兴趣
* 用户点击文章的时间

>总结： 1. 训练集和测试集的用户id没有重复，也就是测试集里面的用户模型是没有见过的
2. 训练集中用户最少的点击文章数是2， 而测试集里面用户最少的点击文章数是1
3. 用户对于文章存在重复点击的情况， 但这个都存在于训练集里面
4. 同一用户的点击环境存在不唯一的情况，后面做这部分特征的时候可以采用统计特征
5. 用户点击文章的次数有很大的区分度，后面可以根据这个制作衡量用户活跃度的特征
6. 文章被用户点击的次数也有很大的区分度，后面可以根据这个制作衡量文章热度的特征
7. 用户看的新闻，相关性是比较强的，所以往往我们判断用户是否对某篇文章感兴趣的时候， 在很大程度上会和他历史点击过的文章有关
8. 用户点击的文章字数有比较大的区别， 这个可以反映用户对于文章字数的区别
9. 用户点击过的文章主题也有很大的区别， 这个可以反映用户的主题偏好 
10. 不同用户点击文章的时间差也会有所区别， 这个可以反映用户对于文章时效性的偏好

### 2、多路召回
1. debug模型：用一小部分数据跑通代码
```
all_user_ids = all_click.user_id.unique()  # nunique->cnt unique->array
sample_user_ids = np.random.choice(all_user_ids, size=100)  # array->array
all_click = all_click[all_click.user_id.isin(sample_user_ids)]  # df->df
```
2. 线下模式：训练调参 `all_click`
3. 线上模式：进行预测，提交结果 `train_click.append(test_click)`

* 

### 3、特征工程
结合用户的历史点击文章信息






