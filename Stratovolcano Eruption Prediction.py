#-*- coding: utf-8 -*-
#How often do stratovolcanoes erupt?

#Importing Libraries

import scipy as sci
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

#Setting the CSV File to a variable

st = pd.read_csv('Stratovolcano Eruption Prediction.csv')
display(st)

st_columns = st[['Zone', 'Height', 'Days Between Eruptions']]

x = st_columns[['Zone', 'Height']]
y = st_columns['Days Between Eruptions']

#Train Test Splitting

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

#Linear Regression

lr = linear_model.LinearRegression()
lr.fit(xtrain, ytrain).predict(xtest).astype(np.int64), ('z' + ' ' + '=' + ' ' + str(lr.coef_[0]) + 'x' + ' ' + str(lr.coef_[1]) + 'y' + ' ' + '+' + ' ' + str(lr.intercept_))
