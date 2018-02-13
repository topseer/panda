# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:12:16 2018

@author: yxu
"""
#import data
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

 
#do some visulization 
crosstable_sepal_len = pd.crosstab(df.sepal_len,y)
crosstable_sepal_len.sum(1).astype(float)

crosstable_sepal_len = crosstable_sepal_len.div(crosstable_sepal_len.sum(1).astype(float),axis = 'index') 
crosstable_sepal_len.plot(kind = "bar",stacked = True)
plt.title('CrossChart_sepal_len')
plt.xlabel('sepal_len')
plt.ylabel('Flower Comibation')
plt.savefig('CrossChart_sepal_len')


crosstable_sepal_wid = pd.crosstab(df.sepal_wid,y)
crosstable_sepal_wid = crosstable_sepal_wid.div(crosstable_sepal_wid.sum(1).astype(float),axis ="index")
crosstable_sepal_wid.plot(kind = "bar",stacked = True)
plt.title = "CrossChart_sepal_wid"
plt.xlabel('sepal_wid')


#standardize the data
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


 
from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = pd.DataFrame(sklearn_pca.fit_transform(X_std))

Y_sklearn = Y_sklearn.join(df["class"])

Y_sklearn = Y_sklearn.rename(index=str, columns={0: "PCA1", 1: "PCA2"})

Y_sklearn["class"] .value_counts()

PCAchart = Y_sklearn.loc[Y_sklearn['class'] =="Iris-setosa"].plot.scatter(x='PCA1', y='PCA2',color = "yellow")
Y_sklearn.loc[Y_sklearn['class'] =="Iris-versicolor"].plot.scatter(x='PCA1', y='PCA2',color = "red", ax = PCAchart)
Y_sklearn.loc[Y_sklearn['class'] =="Iris-virginica"].plot.scatter(x='PCA1', y='PCA2',color = "DarkBlue", ax = PCAchart)




