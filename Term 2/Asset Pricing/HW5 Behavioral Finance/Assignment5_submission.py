# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:31:08 2018

@author: ChanJung Kim
"""


import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt

Num_iterations = 10000
Sim_result_g = np.zeros((Num_iterations, 1))

for i in range(Num_iterations):
    e = float(np.random.standard_normal(1))
    g = np.exp(0.02 + 0.02*e)
    Sim_result_g[i, 0] = g

Rf = 1.0303
b = np.arange(0, 10.001, 0.01)
Sim_result_g = np.squeeze(Sim_result_g)

# function to calculate e(x)
def ex(b0, x):
    R = pd.DataFrame(x*Sim_result_g)
    nu = pd.DataFrame(np.ones(len(Sim_result_g)))
    nu[R>=1.0303]=R-1.0303
    nu[R<1.0303]=2*(R-1.0303)
    ex=0.99*b0*(nu.mean(axis=0).values)+0.99*x-1
    return ex

# Calculating x for each b
Sim_result_x = np.zeros((len(b), 1))

x_data=pd.DataFrame([],columns=['EquilibriumX','Scale Factor'])

for i in range(len(b)):
    x = [1, 1.1]
    e_0=ex(b[i], x[0])
    e_1=ex(b[i], x[1])
    while e_0<0 and e_1>0:
        x_i = 0.5*(x[0]+x[1])
        e_xi=ex(b[i], x_i)
        if abs(e_xi) < 1e-4:
            Sim_result_x[i, 0] = x_i
            break
        elif e_xi<0:
            x[0]=deepcopy(x_i)
        else:
            x[1]=deepcopy(x_i)
    else:
        print('''EquilibriumX initialization window proximity error for 
              b0 = %0.2f. Reinitialize with new window.''' %(i))


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
#plt.savefig('PD_ratio.jpeg', dpi=400)
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
#plt.savefig('Equity Premium.jpeg', dpi=400)
plt.show()
