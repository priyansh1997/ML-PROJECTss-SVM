# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:44:55 2020

@author: PRIYANSH SVM CANCER P
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv(open('cell_samples.csv','rb'))

X = file.iloc[:,[1,2,3,5,6,7,8,9,10]]

X = pd.DataFrame(X)

X.head()

X.tail()

X.columns

X.shape

X.size

X.count()

X['BareNuc'].unique()

X['BareNuc'].value_counts()

X.dtypes

X['BareNuc'] = pd.Series(X['BareNuc'])

X['BareNuc'] = pd.to_numeric(X['BareNuc'], errors='coerce', downcast='integer')

# use either imputer or dorpna to fill the values in place of NaN or to remove the NaN tuple 

#X= X.dropna(subset = ['BareNuc'])

#for removeing all the rows with at least one value NaN

X = X.dropna()

A_df = X.iloc[:,[0,1,2,3,4,5,6,7]]

y = X.iloc[:,8]
"""
#for finding the best features

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(A_df, y, test_size = 0.2)

#from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

classifier = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)


classifier = classifier.fit(X_train, y_train)  

#using print to get the details like Text(170.9,196.385,'X[1] <= 2.5\nentropy = 0.927\

print(tree.plot_tree(classifier.fit(X_train, y_train)))

"""