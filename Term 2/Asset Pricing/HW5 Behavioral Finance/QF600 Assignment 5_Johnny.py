# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:11:47 2018

@author: Johnny

QF600 - Assignment 5
"""

import numpy as np
from scipy.optimize import fsolve,bisect
from matplotlib import pyplot as plt


#### Part 1: HJ Bound
epsilon = np.random.normal(0, 1, 10000)
cons_growth = np.exp(0.02+0.02*epsilon)
delta = 0.99
gamma=1
lambda_p=2
Rf = 1.0303

PD=[]
Rm=[]

b = np.linspace(0,10,100)

for b0 in b:
    x = bisect(lambda x: delta*b0*np.mean([x*c-Rf if (x*c>=Rf) else (2*(x*c-Rf)) for c in cons_growth]) + 0.99 * x - 1 , 1, 1.1)
    PD.append(1/(x-1))
    Rm.append(np.mean(x*cons_growth)-Rf)

plt.plot(b, PD , label = 'PD vs b0')
plt.title('Price-Dividend Ratio over b0')
plt.xlim(left=0)
plt.ylim(bottom=65)
plt.xlabel('b0')
plt.ylabel('P/D')
plt.legend()
plt.grid()
plt.savefig('PD_B0.jpeg', dpi=400)
plt.show()

plt.plot(b, Rm , label = 'EP vs b0')
plt.title('Equity Premium over b0')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel('b0')
plt.ylabel('Equity Premium')
plt.legend()
plt.grid()
plt.savefig('EP_B0.jpeg', dpi=400)
plt.show()

