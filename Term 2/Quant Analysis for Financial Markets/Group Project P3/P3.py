"""
Date: 2018-07-13, 2018-07-14, 2018-11-09
"""

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from numpy.linalg import inv

fn = 'F-F_Research_Data_5_Factors_2x3.CSV'
FF = pd.read_csv(fn, skiprows=3, nrows = 663)
FF = FF.rename(columns={"Unnamed: 0": "Date"})
FF.index = pd.to_datetime(FF.Date, format = '%Y%m').dt.to_period('m')
FF = FF.drop(['Date'], axis=1)
FF = FF*1e-2

fn = '^GSPC.csv'
dji = pd.read_csv(fn)
dji = dji.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
dji.index = pd.to_datetime(dji.Date)
dji = dji.resample('M').last()
dji = dji.drop(['Date'], axis=1)
ri = dji.pct_change()[1:]
ri.index = ri.index.to_period('m')
ri = ri.rename(columns={'Close': 'Return'})

# Align monthly data
df = pd.concat([ri, FF], axis=1)
df = df.dropna()

y = df.Return - df.RF
X = df.iloc[:, 1:2]
# X = np.c_[X, df.iloc[:,4:6]

y = np.asmatrix(y)
y = y.transpose()

X = np.asmatrix(X)
X = np.c_[np.ones(len(y)), X]

Xt = X.transpose()

Z = Xt.dot(X)
Zi = inv(Z)

beta_hat = Zi.dot(Xt.dot(y)) 
print(beta_hat.T)

epsilon = y - X.dot(beta_hat)

RSS = epsilon.transpose().dot(epsilon)
my = np.mean(y)
disp_y = y - my
disp_yt = disp_y.transpose()
TSS = float(disp_yt.dot(disp_y))

R2 = float(1 - RSS/TSS)

n = len(y)
k = len(beta_hat)
vr = RSS/(n-k)

vr = float(vr)
V = Zi*vr

se = np.sqrt(V.diagonal())

t = np.zeros(k, dtype=float)
for i in range(k):
   t[i] = float(beta_hat[i])/se[0,i]

print(t)

# AIC = np.log(vr)+ 2*k/n
# print(AIC)

AIC = n*np.log(RSS/n) + 2*k
print(AIC)





















