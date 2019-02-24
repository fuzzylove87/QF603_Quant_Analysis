"""
Date: 2017-11-23, 2018-02-19
Christopher Ting
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

n = 5000 
theta, alpha = 2.5, 0.5
su2 = 3

np.random.seed(20180219)

u = np.random.normal(0, 1, n)
mu, su = np.mean(u), np.std(u, ddof=1)
u -=  mu
u /- su
u *= np.sqrt(su2)

Y = np.zeros(n, dtype=float)
Y[0] = theta + u[0]
for t in range(1,n):
   Y[t] = theta + alpha * u[t-1] + u[t]

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(Y, lags=20, zero=False, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(Y, lags=20, zero=False, ax=ax2)
plt.show()




