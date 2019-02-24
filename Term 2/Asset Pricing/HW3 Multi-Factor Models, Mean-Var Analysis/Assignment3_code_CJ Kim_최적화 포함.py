# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:53:50 2018

@author: ChanJung Kim
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
from matplotlib import pyplot as plt
import cvxopt as opt
from cvxopt import solvers

FMD = pd.read_excel('Risk_Factors.xlsx', 'Sheet1', parse_dates=['Date'])
IP = pd.read_excel('Industry_Portfolios.xlsx', 'Sheet1', parse_dates=['Date'])
FMD = FMD.iloc[:, 1:]
IP = IP.iloc[:, 1:11]

Rf = pd.DataFrame(FMD['Rf'])
RI_RF = np.subtract(IP, Rf)

# Part1 Performance Measurement
# Sharpe Ratio
Sharpe_R = RI_RF.mean()/RI_RF.std()
print("1. Sharpe Ratio for each industry")
print(Sharpe_R)

# Sortino Ratio
Sortino_RI_RF = RI_RF.copy()
Sortino_RI_RF[Sortino_RI_RF<0] =0
SV = np.mean(np.square(Sortino_RI_RF))
Sortino_R = RI_RF.mean()/SV
print("2. Sortino Ratio for each industry")
print(Sortino_R)

# Jensen's Alpha
RM_RF = np.array(FMD['Rm-Rf'])
RM_RF = sm.add_constant(RM_RF)

def Jensen_Alpha(Y):
    Jensen_Alpha = []
    for i in range(len(Y.columns)):
        CAPM = sm.OLS(Y.iloc[:, i], RM_RF).fit()
        Jensen_Alpha.append(CAPM.params[0])
    return Jensen_Alpha

Jensen_Alpha = pd.DataFrame(Jensen_Alpha(RI_RF))
Jensen_Alpha.index = IP.columns
print("3. Jensen's Alpha for each industry")
print(Jensen_Alpha)

# Three factor's Alpha
TFA_FMD = sm.add_constant(FMD.iloc[:, 1:])

def Three_Factor_Alpha(Y):
    Three_Factor_Alpha = []
    for i in range(len(Y.columns)):
        FAMA = sm.OLS(Y.iloc[:, i], TFA_FMD).fit()
        Three_Factor_Alpha.append(FAMA.params[0])
    return Three_Factor_Alpha

Three_Factor_Alpha = pd.DataFrame(Three_Factor_Alpha(RI_RF))
Three_Factor_Alpha.index = IP.columns
print("3. Three-Factor Alpha for each industry")
print(Three_Factor_Alpha)

# Part2 Minumum Variance
'''
Num_iterations = 100000
Simulation_res = np.zeros((3+len(IP.columns),Num_iterations))
Weights = np.array(np.random.random(len(IP.columns)))
Weights /= np.sum(Weights)
Portfolio_return = np.sum(IP.mean() * Weights)

for i in range(Num_iterations):
    Weights = np.random.random(len(IP.columns))
    Weights /= np.sum(Weights)
    Portfolio_return = np.sum(IP.mean() * Weights)
    Portfolio_std_dev = np.sqrt(np.dot(Weights.T,np.dot(IP.cov(), Weights)))
    Simulation_res[0,i] = Portfolio_return
    Simulation_res[1,i] = Portfolio_std_dev
    Simulation_res[2,i] = Simulation_res[0,i] / Simulation_res[1,i]
    for j in range(len(Weights)):
        Simulation_res[j+3,i] = Weights[j]

Sim_frame = pd.DataFrame(Simulation_res.T, columns = ['Return', 'STD', 'Sharpe', *IP.columns])
# Max_sharpe = Sim_frame.iloc[Sim_frame['Sharpe'].idxmax()]
# Min_std = Sim_frame.iloc[Sim_frame['STD'].idxmin()]
'''
Num_iterations = 250
Simulation_res = np.zeros((3, Num_iterations))

for i in range(Num_iterations):
    Weights = np.random.random(len(IP.columns))
    Weights /= np.sum(Weights)
    Portfolio_return = np.sum(IP.mean() * Weights)
    Portfolio_std_dev = np.sqrt(np.dot(Weights.T,np.dot(IP.cov(), Weights)))
    Simulation_res[0,i] = Portfolio_return
    Simulation_res[1,i] = Portfolio_std_dev
    Simulation_res[2,i] = Simulation_res[0,i] / Simulation_res[1,i]

Sim_frame = pd.DataFrame(Simulation_res.T, columns = ['Return', 'STD', 'Sharpe'])
'''
def frontier(monthly_returns):
    cov = np.matrix(monthly_returns.cov())
    n = monthly_returns.shape[1]
    avg_ret = np.matrix(monthly_returns.mean()).T
    r_min = 0.01
    mus = []
    for i in range(120):
        r_min += 0.01
        mus.append(r_min)
    P = opt.matrix(cov)
    q = opt.matrix(np.zeros((n, 1)))
    G = opt.matrix(np.concatenate((
                -np.transpose(np.array(avg_ret)), 
                -np.identity(n)), 0))
    A = opt.matrix(1.0, (1,n))
    b = opt.matrix(1.0)
    opt.solvers.options['show_progress'] = False
    portfolio_weights = [solvers.qp(P, q, G,
                                    opt.matrix(np.concatenate((-np.ones((1,1))*yy,
                                                               np.zeros((n,1))), 0)), 
                                    A, b)['x'] for yy in mus]
    portfolio_returns = [(np.matrix(x).T * avg_ret)[0,0] for x in portfolio_weights]
    portfolio_stdvs = [np.sqrt(np.matrix(x).T * cov.T.dot(np.matrix(x)))[0,0] for x in portfolio_weights]
    return portfolio_weights, portfolio_returns, portfolio_stdvs

w_f, mu_f, sigma_f = frontier(IP)
'''
plt.style.use('seaborn')
plt.title('Efficient Frontier')
plt.scatter(Sim_frame.STD,Sim_frame.Return,c=Sim_frame.Sharpe,cmap='RdYlGn', edgecolors='black')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()
plt.show()
