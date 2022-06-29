# -*- coding: utf-8 -*-
# @Time : 2021/12/16 15:21
# @Author : li.zhang
# @File : preprocess.py


from utils.pyp import *



# 删除异常值
def find_outliers_by_3segama(dataframe=None, features=None, label=None, verbose=False, is_drop=False):
    # features are numerical type.
    numerical_col = features
    for col in numerical_col:
        col_std = np.std(dataframe[col])
        col_mean = np.mean(dataframe[col])
        outliers_cut_off = col_std * 3
        lower_rule = col_mean - outliers_cut_off
        upper_rule = col_mean + outliers_cut_off
        dataframe[col + '_outliers'] = dataframe[col].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
        if verbose:
            print(dataframe[col + '_outliers'].value_counts())
            print('-'*35)
            print(dataframe.groupby(col + '_outliers')['isDefault'].sum())
            print('='*50)
    if is_drop:
        for col in numerical_col:
            dataframe = dataframe[dataframe[col + '_outliers']=='正常值']
            dataframe = dataframe.reset_index(drop=True)
            dataframe = dataframe.drop(col+'_outliers', axis=1)
    return dataframe

# 特征衍生
def get_mean_of_COL(train_df=None, test_df=None, cols=[], label='label', verbose=False):
    # get mean of col about the label
    for col in cols:
        if verbose:
            print(f'Get mean of {col} about the label.')
        df_dict = train_df.groupby([col])[label].agg(['mean']).reset_index()
        df_dict.index = df_dict[col].values
        dict_col = df_dict['mean'].to_dict()
        train_df[col+'_mean'] = train_df[col].map(dict_col)
        test_df[col+'_mean'] = test_df[col].map(dict_col)
    return train_df, test_df


def get_mean_std_of_CAC(dataframe=None, cols1=[], cols2=[], slience=False):
    # get mean/std of feature about another feature
    for col1 in cols1:
        for col2 in cols2:
            if slience:
                print(f'Get the mean/std. Ex: groupby(\'{col1}\')[\'{col2}\'].transform(\' \')')
            dataframe[col1+'_'+ col2+'_mean'] = dataframe.groupby([col1])[col2].transform('mean')
            dataframe[col1+'_'+ col2+'_std'] = dataframe.groupby([col1])[col2].transform('std')
            dataframe[col1+'_'+ col2+'_mean_c'] = dataframe[col2] - dataframe[col1+'_'+ col2+'_mean']
    return dataframe


# 填充缺失值
def fill_na(train, test):
    numerical_col = list(train.select_dtypes(exclude=['object']).columns)
    category_col = list(filter(lambda x: x not in numerical_col, list(train.columns)))

    train[numerical_col] = train[numerical_col].fillna(train[numerical_col].median())
    train[category_col] = train[category_col].fillna(train[category_col].mode())

    test[numerical_col] = test[numerical_col].fillna(train[numerical_col].median())
    test[category_col] = test[category_col].fillna(train[category_col].mode())
    return train, test


# 特征降维
def pca_pre(tr, n_comp, feat_raw, feat_new):  # TODO will it change the original data
    pca = PCA(n_components=n_comp, random_state=42)
    tr2 = pd.DataFrame(pca.fit_transform(tr[feat_raw]), columns=feat_new)
    return (tr2, pca)

def pca_tra(va, pca_s, feat_raw, feat_new):
    va2 = pd.DataFrame(pca_s.transform(va[feat_raw]), columns=feat_new)
    return va2


# Normalization
def norm_fit(df_1, saveM = True, sc_name='zsco'):
    ss_1_dic = {'zsco':StandardScaler(),
                'mima':MinMaxScaler(),
                'maxb':MaxAbsScaler(),
                'robu':RobustScaler(),
                'norm':Normalizer(),
                'quan':QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1),index = df_1.index,columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2,ss_1)


def norm_tra(df_1,ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1),index = df_1.index,columns = df_1.columns)
    return(df_2)


def one_hot_fit(df_1, fea=None, saveM=True):
    enc = OneHotEncoder(handle_unknown='ignore')
    category_col = fea
    enc.fit(df_1[category_col])

    value = enc.transform(df_1[category_col]).toarray()
    f_name = enc.get_feature_names(category_col)
    df_1[f_name] = value
    df_1 = df_1.drop(category_col, axis=1)
    if saveM == False:
        return (df_1)
    else:
        return (df_1, enc)

def one_hot_tra(df_1, fea=None, oo=None):
    category_col = fea

    value = oo.transform(df_1[category_col]).toarray()
    f_name = oo.get_feature_names(category_col)
    df_1[f_name] = value
    df_1 = df_1.drop(category_col, axis=1)
    return df_1


def lower_sample_data(df, percent=1, tar_name='label'):
    '''
    '''
    data1 = df[df[tar_name] == 1]  # 将多数类别的样本放在data1
    data0 = df[df[tar_name] == 0]  # 将少数类别的样本放在data0
    index = np.random.randint(
        len(data1), size=int(percent * (len(df) - len(data1))))  # 随机给定下采样取出样本的序号
    lower_data1 = data0.iloc[list(index)]  # 下采样
    print(len(data0), len(lower_data1))
    return(pd.concat([lower_data1, data1]))


def sampler(X_train, y_train, how='under', rate=1):
    '''
    and any other
    '''
    X_smote, y_smote = X_train, y_train
    if how == 'over':
        over = RandomOverSampler(sampling_strategy=rate)
        X_smote, y_smote = over.fit_resample(X_train, y_train)
    elif how == 'under':
        under = RandomUnderSampler(sampling_strategy=rate)
        X_smote, y_smote = under.fit_resample(X_train, y_train)
    elif how == 'smote':
        smote = SMOTE(sampling_strategy="auto")
        X_smote, y_smote = smote.fit_resample(X_train, y_train)

    return X_smote, y_smote