# -*- coding: utf-8 -*-
# @Time : 2021/11/16 21:52
# @Author : li.zhang
# @File : preprocessor.py

import pandas as pd
import numpy as np
import datetime


class Preprocessor(object):
    # data preprocess
    def __init__(self, train_data=None, test_data=None):
        self.train = train_data
        self.test = test_data

    def label_encode(self, dataframe):
        #
        # issue-data
        dataframe['issue_date'] = pd.to_datetime(dataframe['issue_date'])
        dataframe['issue_date_y'] = dataframe['issue_date'].dt.year
        dataframe['issue_date_m'] = dataframe['issue_date'].dt.month

        # get the diff bewteen now and origin-date
        origin_date = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
        dataframe['issue_date_diff'] = dataframe['issue_date'].apply(lambda x: x - origin_date).dt.days
        dataframe.drop('issue_date', axis=1, inplace=True)

        # 就业类型
        employer_type = dataframe['employer_type'].value_counts().index
        industry = dataframe['industry'].value_counts().index
        emp_type_dict = dict(zip(employer_type, [0, 1, 2, 3, 4, 5]))
        industry_dict = dict(zip(industry, [i for i in range(15)]))

        # work-year
        dataframe['work_year'].fillna('10+ years', inplace=True)

        work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
                         '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}
        dataframe['work_year'] = dataframe['work_year'].map(work_year_map)

        dataframe['class'] = dataframe['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

        # employer_type and industry
        emp_type_dict = {'政府机构': 1, '幼教与中小学校': 2, '高等教育机构': 3, '世界五百强': 4, '上市企业': 5, '普通企业': 6}
        dataframe['employer_type'] = dataframe['employer_type'].map(emp_type_dict)

        dataframe['industry'] = dataframe['industry'].map(industry_dict)
        return dataframe


    def get_col_year(self, dataframe):
        #
        dataframe[['earlies_credit_1', 'earlies_credit_2']] = dataframe['earlies_credit_mon'].str.split('-',
                                                                                                        expand=True)
        index = (dataframe.earlies_credit_2 >= str('00')) & (dataframe.earlies_credit_2 < str('22'))
        dataframe['earlies_credit_2'][index] = '20' + dataframe['earlies_credit_2'][index]
        index = (dataframe.earlies_credit_2 >= str('22')) & (dataframe.earlies_credit_2 <= str('99'))
        dataframe['earlies_credit_2'][index] = '19' + dataframe['earlies_credit_2'][index]
        index = (dataframe.earlies_credit_2 >= str('1910')) & (dataframe.earlies_credit_2 <= str('2022'))
        dataframe['earlies_credit_2'][~index] = '2000'

        # earlies_credit_mon
        dataframe['earlies_credit_2'] = pd.to_datetime(dataframe.earlies_credit_2)
        origin_date = datetime.datetime.strptime('1920-06-01', '%Y-%m-%d')
        dataframe['earlies_credit_diff'] = dataframe['earlies_credit_2'].apply(lambda x: x - origin_date).dt.days
        dataframe.drop(['earlies_credit_mon', 'earlies_credit_1', 'earlies_credit_2'], axis=1, inplace=True)
        return dataframe

    def find_outliers_by_3segama(self, dataframe=None, features=None, label=None, verbose=False, is_drop=False):
        # features are numerical type.
        numerical_col = features
        for col in numerical_col:
            col_std = np.std(dataframe[col])
            col_mean = np.mean(dataframe[col])
            outliers_cut_off = col_std * 3
            lower_rule = col_mean - outliers_cut_off
            upper_rule = col_mean + outliers_cut_off
            dataframe[col + '_outliers'] = dataframe[col].apply(
                lambda x: str('异常值') if x > upper_rule or x < lower_rule else '正常值')
            if verbose:
                print(dataframe[col + '_outliers'].value_counts())
                print('-' * 35)
                print(dataframe.groupby(col + '_outliers')['isDefault'].sum())
                print('=' * 50)
        if is_drop:
            for col in numerical_col:
                dataframe = dataframe[dataframe[col + '_outliers'] == '正常值']
                dataframe = dataframe.reset_index(drop=True)
                dataframe = dataframe.drop(col + '_outliers', axis=1)
        return dataframe

    def get_mean_of_COL(self, train_df=None, test_df=None, cols=[], label='label', verbose=False):
        # get mean of col about the label
        for col in cols:
            if verbose:
                print(f'Get mean of {col} about the label.')
            df_dict = train_df.groupby([col])[label].agg(['mean']).reset_index()
            df_dict.index = df_dict[col].values
            dict_col = df_dict['mean'].to_dict()
            train_df[col + '_mean'] = train_df[col].map(dict_col)
            test_df[col + '_mean'] = test_df[col].map(dict_col)
        return train_df, test_df

    def drop_cols(self, dataframe=None, cols=[]):
        return dataframe.drop(cols,
                              axis=1)  # 'loan_id', 'user_id', 'policy_code',  'scoring_low', 'scoring_high', 'f1', 'early_return_amount_3mon'

    def get_mean_std_of_CAC(self, dataframe=None, cols1=[], cols2=[], slience=False):
        # get mean/std of feature about another feature
        for col1 in cols1:
            for col2 in cols2:
                if slience:
                    print(f'Get the mean/std. Ex: groupby(\'{col1}\')[\'{col2}\'].transform(\' \')')
                dataframe[col1 + '_' + col2 + '_mean'] = dataframe.groupby([col1])[col2].transform('mean')
                dataframe[col1 + '_' + col2 + '_std'] = dataframe.groupby([col1])[col2].transform('std')
        return dataframe