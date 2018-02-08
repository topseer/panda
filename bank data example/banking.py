# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:19:00 2018

@author: yxu
"""

import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("C:/YangXu/panda/bank data example/banking.csv")

data = data.dropna()

data.head()

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    #var = 'job'
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
    


cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values
    
data_final_vars=data_final.columns.values.tolist()

y=['y']
X=[i for i in data_final_vars if i not in y]


logreg = LogisticRegression()

rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )

print(rfe.support_)
print(rfe.ranking_)

newX = list([]);
for i,element in enumerate(rfe.support_):
  if element:
    print (i)
    newX.append(X[i])


import statsmodels.api as sm
from sklearn.cross_validation import train_test_split

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())


X_data=data_final[newX]
y_data=data_final['y']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)


logreg.score(X_test, y_test)




from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))




