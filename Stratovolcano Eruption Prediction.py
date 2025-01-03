# -*- coding: utf-8 -*-
"""Stratovolcano Eruption Prediction

i = 1
j = 1
a = 0
b = 0
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
st = pd.read_csv('Stratovolcano.csv')

while i < len(st):
  if ((st['Last Eruption Date'][i][-2:]) + (st['Second To Last Eruption Date'][i][-2:])) == 'BCBC':
    lst.append(((int(365.25 * ((int(st['Second To Last Eruption Date'][i][:-9])) - (int(st['Last Eruption Date'][i][:-9])))))) + (int((30.4375 * ((int(st['Second To Last Eruption Date'][i][-8:-6])) - (int(st['Last Eruption Date'][i][-8:-6])))))) + (int(st['Second To Last Eruption Date'][i][-5:-3])) - (int(st['Last Eruption Date'][i][-5:-3])))
    i += 1
  elif (st['Second To Last Eruption Date'][i][-2:]) == 'BC':
    (((int(365.25 * ((int(st['Last Eruption Date'][i][:-6])) + (int(st['Second To Last Eruption Date'][i][:-9])))))) + (int((30.4375 * ((int(st['Last Eruption Date'][i][-5:-3])) + (int(st['Second To Last Eruption Date'][i][-8:-6])))))) + (int(st['Last Eruption Date'][i][-2:])) + (int(st['Second To Last Eruption Date'][i][-5:-3])))
    i += 1
  else:
    lst.append(((int(365.25 * ((int(st['Last Eruption Date'][i][:-6])) - (int(st['Second To Last Eruption Date'][i][:-6])))))) + (int((30.4375 * ((int(st['Last Eruption Date'][i][-5:-3])) - (int(st['Second To Last Eruption Date'][i][-5:-3])))))) + (int(st['Last Eruption Date'][i][-2:])) - (int(st['Second To Last Eruption Date'][i][-2:])))
    i += 1

st['Days Between Eruptions'] = lst
stv = st[['Zone', 'Days Between Eruptions']]

x = np.array(stv.iloc[:, :-1][1:len(stv)]).reshape(-1,1)
y = np.array(stv.iloc[:, -1:][1:len(stv)]).reshape(-1,1)

while j <= len(y):
  ylst.append(y[(j - 1):j][0][0])
  j += 1

y = np.array(ylst)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.143)

xtest = np.array([[2000], [8000]])

clf = linear_model.LinearRegression()
clf.fit(xtrain, ytrain).predict(xtest).astype(np.int64), str(clf.coef_[0]) + 'x' + ' ' + '+' + ' ' + str(clf.intercept_)

while a < len(xtrain):
  xl.append(xtrain[a][0])
  a += 1

while b < len(ytrain):
  yl.append(ytrain[b])
  b += 1

coef = np.polyfit(xl, yl, 1)
coeff = np.poly1d(coef)
plt.plot(xl, yl, 'ko', x, coeff(x), 'k')
plt.title('Stratovolcano Eruption Predictions')
plt.xlabel('Zone')
plt.ylabel("Days Between Eruptions")