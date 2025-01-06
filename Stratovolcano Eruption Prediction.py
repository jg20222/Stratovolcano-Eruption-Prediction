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

while i < len(st):
  if ((st['Last Eruption Date'][i][-2:]) + (st['Second To Last Eruption Date'][i][-2:])) == 'BCBC':
    lst.append(((int(365.25 * ((int(st['Second To Last Eruption Date'][i][:-9])) - (int(st['Last Eruption Date'][i][:-9])))))) + (int((30.4375 * ((int(st['Second To Last Eruption Date'][i][-8:-6])) - (int(st['Last Eruption Date'][i][-8:-6])))))) + (int(st['Second To Last Eruption Date'][i][-5:-3])) - (int(st['Last Eruption Date'][i][-5:-3])))
    i += 1
  elif (st['Second To Last Eruption Date'][i][-2:]) == 'BC':
    lst.append(((int(365.25 * ((int(st['Last Eruption Date'][i][:-6])) + (int(st['Second To Last Eruption Date'][i][:-9])))))) + (int((30.4375 * ((int(st['Last Eruption Date'][i][-5:-3])) + (int(st['Second To Last Eruption Date'][i][-8:-6])))))) + (int(st['Last Eruption Date'][i][-2:])) + (int(st['Second To Last Eruption Date'][i][-5:-3])))
    i += 1
  else:
    lst.append(((int(365.25 * ((int(st['Last Eruption Date'][i][:-6])) - (int(st['Second To Last Eruption Date'][i][:-6])))))) + (int((30.4375 * ((int(st['Last Eruption Date'][i][-5:-3])) - (int(st['Second To Last Eruption Date'][i][-5:-3])))))) + (int(st['Last Eruption Date'][i][-2:])) - (int(st['Second To Last Eruption Date'][i][-2:])))
    i += 1

st['Days Between Eruptions'] = lst
stv = st[['Zone', 'Days Between Eruptions']]

x = np.array(stv.iloc[:, :-1][1:len(stv)]).reshape(-1,1)
y = np.array(stv.iloc[:, -1:][1:len(stv)]).reshape(-1,1)

while a <= len(y):
  ylst.append(y[(a - 1):a][0][0])
  a += 1

y = np.array(ylst)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 1/((len(stv)) - 3))

xtest = np.array([[2000], [8000]])

clf = linear_model.LinearRegression()
clf.fit(xtrain, ytrain).predict(xtest).astype(np.int64), (str(clf.coef_[0]) + 'x' + ' ' + '+' + ' ' + str(clf.intercept_))

while b < len(x):
  xl.append(x[b][0])
  b += 1

while c < len(y):
  yl.append(y[c])
  c += 1

coef = np.polyfit(xl, yl, 1)
coeff = np.poly1d(coef)
plt.plot(x, y, 'ko', x, coeff(x), 'k')
plt.title('Stratovolcano Eruption Predictions')
plt.xlabel('Zone')
plt.ylabel("Days Between Eruptions In Millions")