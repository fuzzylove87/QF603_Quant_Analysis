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

def Pi(tau, S, K, sigma,r):
    T_sqrt = np.sqrt(tau)
    d1 = (np.log(S/K)+(r-sigma**2/2)*tau) / (sigma*T_sqrt)
    pi = K*np.exp(-2*r*tau)*norm.cdf(d1)
#    pi = K*norm.cdf(d1)
    return pi

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

Delta_Difference = D[:, 1:] - D[:, :-1]

# Delta Difference * Stock price
Rebalanced_S = Delta_Difference*S[:, 1:]

# Define Bond
B = np.zeros([i,N+1])

for y in range(0,i+1):
    for x in range(len(steps)):
        B[y-1, x]= np.exp(r*steps[x])
    
Pi_B = np.zeros([i,N+1])
    
for y in range(0,i):
    for x in range(len(steps)):
        Pi_B[y, x]= Pi(1/12 - steps[x], S[y, x], K, sigma, r)

Pi_Difference = Pi_B[:, 1:] - Pi_B[:, :-1]

# Pi Difference * Bond price
Rebalanced_B = Pi_Difference*B[:, 1:]

PL = Rebalanced_S - Rebalanced_B

# Int rate
int_steps = np.linspace(T, 0, N)

Intrate = np.zeros([i,N])
Intrate[:, -1] = 1

for y in range(0,i):
    for b in range(len(int_steps)):
        Intrate[y, b] = np.exp(r*int_steps[b])

Final_PL = PL*Intrate
Sum_Final_PL = np.sum(Final_PL, axis=1)

Sum_Final_PL = pd.DataFrame(Sum_Final_PL)
Sum_Final_PL.hist(bins=50)
plt.title('Histogram of the hedging error')
plt.xlabel('Final Profit and Loss')
plt.ylabel('Frequency')
#plt.savefig('N=84.jpeg', dpi=600)

'''
D = pd.DataFrame(np.sum(D, axis=1))
Delta_Difference = pd.DataFrame(np.sum(Delta_Difference, axis=1))
Pi_B = pd.DataFrame(np.sum(Pi_B, axis=1))
Pi_Difference = pd.DataFrame(np.sum(Pi_Difference, axis=1))
Intrate  = pd.DataFrame(np.sum(Intrate, axis=1))

D.hist(bins =100)
Delta_Difference.hist(bins = 100)
Pi_B.hist(bins =100)
Pi_Difference.hist(bins =100)
Intrate.hist(bins =100)
'''




