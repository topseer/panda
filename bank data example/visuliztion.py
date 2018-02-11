# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:19:00 2018

@author: yxu
"""

import pandas as pd
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt 

data = pd.read_csv("C:/panda/bank data example/banking.csv")

msg = "test"

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



data_final["y"].value_counts()



a = sns.countplot(x = "y", data = data_final,palette="hls")
 

jobtable = pd.crosstab(data.job,data.y)
jobtable = jobtable.div(jobtable.sum(1).astype(float),axis = 'index') 
jobtable.plot(kind = "bar",stacked = True)


 


plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')


table=pd.crosstab(data.marital,data.y)
table.plot(kind='bar', stacked=True)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')

