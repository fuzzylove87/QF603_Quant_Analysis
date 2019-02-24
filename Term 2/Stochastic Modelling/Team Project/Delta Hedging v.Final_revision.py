# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:19:13 2018

@author: ykfri
"""

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define basic function
def BlackScholes(tau, S, K, sigma,r):
    d1=(np.log(S/K)+(r+(sigma**2)/2)*tau)/(sigma*np.sqrt(tau))
    d2=d1-sigma*np.sqrt(tau)
    c=(S*norm.cdf(d1)-K*np.exp(-r*tau)*norm.cdf(d2))
    return c

def Delta(tau, S, K, sigma, r):
    T_sqrt = np.sqrt(tau)
    d1 = (np.log(S/K)+(r+sigma**2/2)*tau) / (sigma*T_sqrt)
    Delta = norm.cdf(d1)
    return Delta

#Simulation of Stock Price

S0 = 100 #initial stock price
K = 100 #strike price
r = 0.05 #risk-free interest rate
sigma = 0.20 #volatility in market
T = 1/12 #time in years
N = 21 #number of steps within each simulation
deltat = T/N #time step
i = 5000 #number of simulations
discount_factor = np.exp(-r*T) #discount factor

S = np.zeros([i,N+1])
t = range(0,N+1,1)

for y in range(0,i):
    S[y,0]=S0
    for x in range(0,N):
        S[y,x+1] = S[y,x]*(np.exp((r-(sigma**2)/2)*deltat + sigma*np.sqrt(deltat)*np.random.normal(0,1)))
    plt.plot(t,S[y])    

plt.title('Simulations %d Steps %d Sigma %.2f r %.2f S0 %.2f' % (i, N, sigma, r, S0))
plt.xlabel('Steps')
plt.ylabel('Stock Price')
plt.show()

#Define Delta
steps = np.linspace(0, T, N+1)

D = np.zeros([i,N+1])

for y in range(0,i):
    for x in range(len(steps)):
        D[y,x] = Delta(1/12 - steps[x], S[y, x], K, sigma, r)

# Int rate
int_steps = np.linspace(T, 0, N+1)

Intrate = np.zeros([i,N+1])
Intrate[:, -1] = 1

for y in range(0,i):
    for b in range(len(int_steps)):
        Intrate[y, b] = np.exp(r*int_steps[b])

PL = np.zeros((i, N+1))
PL[:, 0] = BlackScholes(T, S0, K, sigma, r) - D[0, 0]*S0

for y in range(0,i):
    for x in range(len(steps)-1):
        PL[y, x+1] = (Delta(T-steps[x], S[y, x], K, sigma, r)-Delta(T-steps[x+1], S[y, x+1], K, sigma, r))*S[y, x+1]

PL[:, -1] = D[:, -2]*S[:, -1] - np.maximum(S[:, -1]-K, 0)
Final_PL2 = PL*Intrate
Sum_Final_PL = np.sum(Final_PL2, axis=1)
Sum_Final_PL = pd.DataFrame(Sum_Final_PL)
Sum_Final_PL.hist(bins=50)
plt.title('Histogram of the hedging error')
plt.xlabel('Final Profit and Loss')
plt.ylabel('Frequency')
#plt.savefig('N_21.jpeg')