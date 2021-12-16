## models

### DCNN
* DCNN网络定义 `DCNN(nn.Module)`

### DNN
* DNN网络定义 `DNN(nn.Module)`

## evaluate

### baseline
* `LogisticRegression`
* `KNeighborsClassifier`
* `RandomForestClassifier`
* `tree.DecisionTreeClassifier()`
* `svm.SVC()`
* `XGBoost`
* `LightGBM`
* `CatBoost`

### lightgbm_eval

* n折交叉验证 `lgb_train_eval(train_set=None, tar_name='label', ratio=1, test_set=None, is_test=False, thd=0.5)`
* 学习曲线 `adjust_lgb_param(train_set=None, tar_name='label', test_set=None, metric='accuracy')`
* 学习曲线对比 `adjust_lgb_parameters(train_set=None, tar_name='label', test_set=None)`
* 网格搜索 `GridSearchCV_lgb(train_set=None, tar_name='label')`
* 随机搜索 `RandomizedSearchCV_lgb(train_set=None, tar_name='label')`
* 贝叶斯搜索 `bayes_lgb(train_set=None, tar_name='label')`

### xgboost_eval
* `None`

### catboost_eval
* `None`

### DCDD_eval
* 1DCNN训练与验证 `d_cnn_train_eval(train=None, test=None, tar_name='label')`

### DNN
* DNN训练与验证 `dnn_train_eval(train=None, test=None, tar_name='label')`

## utils

### common_tools

* 分段统计函数 `segment_statistic(y_test=None, prob_y=None, bins=None)`
* 二分类评估指标 `binary_classifier_metrics(test_labels, predict_labels, predict_prob, show_flag=True)`
* 特征重要性 `show_feature_importance` 
* 特征重要性可视化 `plt_feature_importance` 
* 相关系数热力图 `plot_heatmap` 
* AUC可视化 `create_roc`

### visualization

* 连续变量正负样本展示 `plot_feature_kde(train_data, test_data, features=[])`
* 正负样本展示 `show_pie()`

### preprocess

* 读取数据 `read_data(name=None, source_path=None)`
* 数据概述 `describe_data(train, test)`
* 删除异常值 `find_outliers_by_3segama(dataframe=None, features=None, label=None, verbose=False, is_drop=False)`
* 特征衍生 `get_mean_of_COL(train_df=None, test_df=None, cols=[], label='label', verbose=False)`
* 特征衍生 `get_mean_std_of_CAC(dataframe=None, cols1=[], cols2=[], slience=False)`
* 缺失值填充 `fill_na(train, test)`
* 数据降维 `pca_pre(tr, n_comp, feat_raw, feat_new)` & `pca_tra(va, pca_s, feat_raw, feat_new)`
* 标准化 `norm_fit(df_1, saveM = True, sc_name='zsco')` & `norm_tra(df_1,ss_x)`


### tricks
* 训练集和测试集是否可分 `test_dataset(origin_train=None, origin_test=None)`

### nn_tools
* 定义损失函数 `SmoothBCEwLogits(_WeightedLoss)`
* pytorch训练集 `TrainDataset()` 
* pytorch测试集 `TestDataset()`


## main

## TODO
* 采样函数
* 可视化
* 模型集成
