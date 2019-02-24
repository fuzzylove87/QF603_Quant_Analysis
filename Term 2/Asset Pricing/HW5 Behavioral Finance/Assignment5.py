# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:31:08 2018

@author: ChanJung Kim
"""


import numpy as np
from scipy.optimize import bisect, fsolve
from matplotlib import pyplot as plt

Num_iterations = 10000
Sim_result_g = np.zeros((Num_iterations, 1))

for i in range(Num_iterations):
    e = float(np.random.standard_normal(1))
    g = np.exp(0.02 + 0.02*e)
    Sim_result_g[i, 0] = g

Rf = 1.0303
b = np.arange(0, 10.001, 0.1)

# function to calculate e(x)
def ex(x, b):
    if x*Sim_result_g.mean() >= Rf:
        e_x = 0.99*b*(x*Sim_result_g.mean()-Rf)+0.99*x-1
    elif x*Sim_result_g.mean() < Rf:
        e_x = 0.99*b*2*(x*Sim_result_g.mean()-Rf)+0.99*x-1
    return e_x

# Calculating x for each b
Sim_result_x = np.zeros((len(b), 1))

a = 0
x_negative = 1
x_positive = 1.1

for b0 in b:
    x = bisect(ex, x_positive, x_negative, args=(b0))
    Sim_result_x[a, 0] = x
    a +=1

print(fsolve(ex, 1, args=(0)))
print(fsolve(ex, 1, args=(10)))

# Price-Dividend Ratio
pd_ratio = 1/(Sim_result_x-1)
pd_ratio = np.squeeze(pd_ratio)

plt.title('Price-Dividend Ratio for Given b0')
plt.plot(b,pd_ratio, label ='Price-Dividend Ratio')
plt.legend()
plt.grid()
plt.xlim(0, 10)
plt.xlabel('b0', size=10)
plt.ylabel('Price-Dividend Ratio', size=10)
plt.savefig('PD_ratio.jpeg', dpi=400)
plt.show()

# Equity Premium
Market_Return = Sim_result_x*Sim_result_g.mean()
Equity_Premium = Market_Return - Rf

plt.title('Equity Premium for Given b0')
plt.plot(b, Equity_Premium, label ='Equity Premium')
plt.legend()
plt.grid()
plt.xlim(0, 10)
plt.xlabel('b0', size=10)
plt.ylabel('Equity Premium', size=10)
plt.savefig('Equity Premium.jpeg', dpi=400)
plt.show()



