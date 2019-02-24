# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:14:05 2018

@author: Brandon Chan
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

def Pi(tau, S, K, sigma,r):
    T_sqrt = np.sqrt(tau)
    d1 = (np.log(S/K)+(r-sigma**2/2)*tau) / (sigma*T_sqrt)
    pi = K*np.exp(-2*r*tau)*norm.cdf(d1)
    return pi

#Simulation of Stock Price

S0 = 100 #initial stock price
K = 100 #strike price
r = 0.05 #risk-free interest rate
sigma = 0.20 #volatility in market
T = 1/12 #time in years
N = 21 #number of steps within each simulation
deltat = T/N #time step
i = 1 #number of simulations
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

#Define bond
steps = np.linspace(T, 0, N)
B = np.zeros([1,N])
Intrate = np.zeros([1,N])

for b in range(len(steps)):
    B[0, b] = Bond(steps[b], S0, K, sigma, r)
    
for b in range(len(steps)-1, 0, -1):
    Intrate[0, b] = np.exp(r*steps[b])

#Delta Hedging
#Define St-K steps
C = S -K
C[C<0] = 0

#Define delta
D = np.zeros([i,N])

for y in range(0,i):
    for x in range(len(steps)):
        D[y,x] = Delta(steps[x], S[y, x], K, sigma, r) 

# B_rebalance = D*S - B

BS = np.zeros([i,N])

for y in range(0,i):
    for x in range(0,N):
        BS[y,x] = BlackScholes(steps[x], S[y, x], K, sigma,r)
        
AAAA = B_rebalance - BS
KKK = AAAA * Intrate





Profit_n_Loss=np.sum(math.exp**(r*deltat)*(BlackScholes(deltat, 100, 100, 0.20,0.05)-C))
Profit_n_Loss=BlackScholes(deltat, 100, 100, 0.20,0.05)-C
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






