# -*- coding: utf-8 -*-
"""Stratovolcano Eruption Prediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wRMjw6QJmS_lm4w23pANJHzlwTj7KCjq
"""

i = 1
a = 1
b = 0
c = 0
lst = [4392130]
xl = []
yl = []
ylst = []

import scipy as sci
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

'How often do stratovolcanoes erupt?'
st = pd.read_csv('Stratovolcano Eruption Prediction.csv')

stv = st[['Zone', 'Height', 'Days Between Eruptions']]

x = stv[['Zone', 'Height']]
y = stv['Days Between Eruptions']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

lr = linear_model.LinearRegression()
lr.fit(xtrain, ytrain).predict(xtest).astype(np.int64), ('z' + ' ' + '=' + ' ' + str(lr.coef_[0]) + 'x' + ' ' + str(lr.coef_[1]) + 'y' + ' ' + '+' + ' ' + str(lr.intercept_))