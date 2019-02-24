# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:14:05 2018

@author: Brandon Chan
"""


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

#Simulation of Stock Price

S0 = 100 #initial stock price
K = 100 #strike price
r = 0.05 #risk-free interest rate
sigma = 0.20 #volatility in market
T = 1/12 #time in years
N = 84 #number of steps within each simulation
deltat = T/N #time step
i = 5000 #number of simulations
discount_factor = np.exp(-r*T) #discount factor

S = np.zeros([i,N])
t = range(0,N,1)

for y in range(0,i-1):
    S[y,0]=S0
    for x in range(0,N-1):
        S[y,x+1] = S[y,x]*(np.exp((r-(sigma**2)/2)*deltat + sigma*np.sqrt(deltat)*np.random.normal(0,1)))
    plt.plot(t,S[y])


plt.title('Simulations %d Steps %d Sigma %.2f r %.2f S0 %.2f' % (i, N, sigma, r, S0))
plt.xlabel('Steps')
plt.ylabel('Stock Price')
plt.show()


#Delta Hedging

C = np.zeros((i-1,1))
for y in range(0,i-1):
    C[y]=np.maximum(S[y,N-1]-K,0)

'''
CallPayoffAverage = np.average(C)
CallPayoff = discount_factor*CallPayoffAverage
print(CallPayoff)
'''

def BlackScholes(tau, S, K, sigma,r):
    d1=(np.log(S/K)+(r+(sigma**2)/2)*tau)/(sigma*np.sqrt(tau))
    d2=d1-sigma*np.sqrt(tau)
    c=(S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d2))
    return c


#Profit_n_Loss=np.sum(math.exp**(r*deltat)*(BlackScholes(deltat, 100, 100, 0.20,0.05)-C))
steps = np.linspace(0, T, N)
Profit_n_Loss = np.zeros((N,1))

for a in range(len(steps)):
    Profit_n_Loss[a, 0] = BlackScholes(steps[a], 100, 100, 0.20,0.05)-C[a, 0]
    
'''
Profit_n_Loss=pd.DataFrame(Profit_n_Loss,columns=['Final P&L'])
PL=Profit_n_Loss-Profit_n_Loss.shift()
PL=PL.dropna()
PL = pd.DataFrame(PL)
PL.hist(bins=30)
plt.xlabel('Final Profit and Loss')
plt.ylabel('Frequency')


print("----------------------------------------------------------------------")
print("Mean of P&L")
print(np.mean(PL))
print("----------------------------------------------------------------------")
print("Standard Deviation of P&L")
print(np.std(PL))

'''





