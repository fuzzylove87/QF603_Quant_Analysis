# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:12:54 2018

@author: ChanJung Kim
"""

import numpy as np
from matplotlib import pyplot as plt

Num_iterations = 10000
Sim_result_g = np.zeros((Num_iterations, 1))

for i in range(Num_iterations):
    rv = np.random.random()
    e = float(np.random.standard_normal(1))
    if rv < 0.983:
        nu = 0
    elif rv >= 0.983:
        nu = np.log(0.65)
    g = np.exp(0.02 + 0.02*e + nu)
    Sim_result_g[i, 0] = g          


# Part 1
gamma = np.arange(1, 4.001, 0.01)
Sim_result_HJ = np.zeros((len(gamma), 1))
a = 0

for i in gamma:
    M = 0.99*Sim_result_g**(-i)
    mean = M.mean()
    sd = M.std()
    Sim_result_HJ[a, 0] = sd/mean
    a += 1

Sim_result_HJ = np.squeeze(Sim_result_HJ)

for i in range(len(Sim_result_HJ)):
    if Sim_result_HJ[i] > 0.4:
        k = i
        break
#    elif Sim_result_HJ[i] < 0.4:
#        break

print('Smallest value of gamma for which H-J bound > 0.4 is %0.4f' %(gamma[k]))
        
plt.title('Hansen-Jagannathan Bound for Given Gamma')
plt.plot(gamma,Sim_result_HJ, label ='Hansen-Jagannathan Bound')
plt.scatter(gamma[k], Sim_result_HJ[k], color = 'r', s=25, label = 'The point where SD(M)/E(M) exceeds 0.4')
plt.legend()
plt.grid()
plt.xlim(1, 4)
plt.xlabel('Gamma', size=10)
plt.ylabel('Hansen-Jagannathan Bound', size=10)
#plt.savefig('Hansen_Jagannathan_Bound.jpeg', dpi=400)
plt.show()

# Part 2
gamma = np.arange(1, 7.001, 0.01)
Sim_result_PD = np.zeros((len(gamma), 1))

a = 0
for i in gamma:
    PD = 0.99*Sim_result_g**(1-i)
    pd_mean = PD.mean()
    Sim_result_PD[a, 0] = pd_mean
    a += 1

Sim_result_PD = np.squeeze(Sim_result_PD)    

plt.title('Price-Dividend Ratio for Given Gamma')
plt.plot(gamma,Sim_result_PD, label ='Price-Dividend Ratio')
plt.legend()
plt.grid()
plt.xlim(1, 7)
plt.xlabel('Gamma', size=10)
plt.ylabel('Price-Dividend Ratio', size=10)
#plt.savefig('Price_Dividend_Ratio.jpeg', dpi=400)
plt.show()

# Part 3
plt.title('Equity Premium for Given Gamma')
Rm = 1/Sim_result_PD * Sim_result_g.mean()

Sim_result_RF = np.zeros((len(gamma), 1))
a = 0

for i in gamma:
    De = 0.99*Sim_result_g**(-i)
    Rf = 1/De.mean()
    Sim_result_RF[a, 0] = Rf
    a += 1

Sim_result_RF = np.squeeze(Sim_result_RF) 
RP = Rm - Sim_result_RF

plt.plot(gamma,RP, label ='Equity Premium')
plt.legend()
plt.grid()
plt.xlim(1, 7)
plt.xlabel('Gamma', size=10)
plt.ylabel('Equity Premium', size=10)
#plt.savefig('Equity_Premium.jpeg', dpi=400)
plt.show()




