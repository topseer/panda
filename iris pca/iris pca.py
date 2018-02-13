# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:12:16 2018

@author: yxu
"""

import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()



X = df.ix[:,0:4].values
y = df.ix[:,4].values