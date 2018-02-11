# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:35:42 2018

@author: Yang Xu
"""

import pandas as pd
import numpy as np


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')



chipo.info()
list(chipo)

chipo.columns

chipo.index



chipo.groupby('item_name').count()
chipo.groupby('item_name').sum()


removeDollar = lambda  x: float(x[1:])

#remove     
chipo["item_price_float"]=chipo["item_price"].apply (removeDollar)

chipo["item_price_float"].head()


string = "afdlklfd"

string[1:]


chipo.order_id.count()

chipo.order_id.value_counts().count()



chipo.head()


avg = chipo.groupby('order_id').sum()["item_price_float"]
avg = avg.rename ("price avg")

avg = avg.rename(["item_price_float","avg"])

avg.head()

price_sum = chipo.groupby('order_id').mean()["item_price_float"]
price_sum = price_sum.rename ("price sum")


 
pd.concat(
        [avg,
         price_sum
        ],
        join = "inner"        ,
        axis = 1
        )



chipo.groupby('item_name').count()
