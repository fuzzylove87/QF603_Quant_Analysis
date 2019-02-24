"""
Session 1 Assignment on Returns
Christopher Ting
2018-04-03, 2018-06-19
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_excel('MSFT.xlsx')
# plt.plot(df['Actual Value'])
# plt.show()

n = len(df)
i = range(3, n, 4)
df3 = df.iloc[i]
m = len(df3)
q = range(m)
df3.index = q
q = np.asarray(q)


plt.plot(df3['Actual Value'], '-ro', )
plt.show()

y = np.asmatrix(df3['Actual Value'])

e = 1
X = [np.ones(m),  q**e]
Xprime = np.asmatrix(X)
X = Xprime.transpose()

XpX = Xprime.dot(X)
XpXi = np.linalg.inv(XpX)
beta = XpXi.dot(Xprime.dot(y.transpose()))
new_q = len(q)
predicted_EPS = beta[0] + beta[1]*new_q**e
print("OLS Prediction of EPS = %0.2f" % (predicted_EPS))

XX = (np.matrix(X)*np.matrix(beta)).reshape(1, -1)
RR = y - XX
TSS = np.sum(np.square(y - y.mean()))
RSS = np.sum(np.square(RR))
R_square = 1 - RSS/TSS

t = stats.t.ppf(0.975, 23)
sigma = np.sqrt(RSS/(24-2))
inner_no = (new_q**e-X[:,1].mean())**2
inner_deno = np.sum(np.square(X[:,1]-X[:,1].mean()))


aa = t*sigma*np.sqrt(1+(1/25)+inner_no/inner_deno)

print(df3)
print(TSS)
print(RSS)
print(R_square)
print(predicted_EPS-aa)
print(predicted_EPS+aa)



