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
import scipy.optimize as sco

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

plt.title('Sharpe Ratio', size=20)
plt.bar(Sharpe_R.index, Sharpe_R, facecolor='#9999ff', edgecolor='white')
plt.xlabel('Industry', size=15)
plt.ylabel('Sharpe Ratio', size=15)
plt.ylim(0.00, 0.30)
for x, y in zip(Sharpe_R.index, Sharpe_R):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x , y+0.02, '%.2f' % y, ha='center', va='top')
plt.show()

# Sortino Ratio
Sortino_RI_RF = RI_RF.copy()
Sortino_RI_RF[Sortino_RI_RF>0] =0
SV = (np.mean(np.square(Sortino_RI_RF)))*(len(Sortino_RI_RF)/(len(Sortino_RI_RF)-1))
Sortino_R = RI_RF.mean()/np.sqrt(SV)
print("2. Sortino Ratio for each industry")

plt.title('Sortino Ratio', size=20)
plt.bar(Sortino_R.index, Sortino_R, facecolor='#9999ff', edgecolor='white')
plt.xlabel('Industry', size=15)
plt.ylabel('Sortino Ratio', size=15)
plt.ylim(0.00, 0.4)
for x, y in zip(Sortino_R.index, Sortino_R):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x , y+0.02, '%.2f' % y, ha='center', va='top')
plt.show()


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
Jensen_Alpha = np.squeeze(Jensen_Alpha)
print("3. Jensen's Alpha for each industry")
print(Jensen_Alpha)

plt.title('Jensen\'s Alpha', size=20)
plt.bar(Jensen_Alpha.index, Jensen_Alpha, facecolor='#9999ff', edgecolor='white')
plt.xlabel('Industry', size=15)
plt.ylabel('Jensen\'s Alpha', size=15)
plt.ylim(-0.5, 0.6)
for x, y in zip(Jensen_Alpha.index, Jensen_Alpha):
    if y >= 0:
        plt.text(x , y+0.05, '%.2f' % y, ha='center', va='top')
    elif y < 0:
        plt.text(x , y-0.05, '%.2f' % y, ha='center', va='bottom')
plt.show()

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
Three_Factor_Alpha = np.squeeze(Three_Factor_Alpha)

print("3. Three-Factor Alpha for each industry")
print(Three_Factor_Alpha)

plt.title('Three-Factor Alpha', size=20)
plt.bar(Three_Factor_Alpha.index, Three_Factor_Alpha, facecolor='#9999ff', edgecolor='white')
plt.xlabel('Industry', size=15)
plt.ylabel('Three_Factor_Alpha', size=15)
plt.ylim(-0.6, 0.6)
for x, y in zip(Three_Factor_Alpha.index, Three_Factor_Alpha):
    if y >= 0:
        plt.text(x , y+0.05, '%.2f' % y, ha='center', va='top')
    elif y < 0:
        plt.text(x , y-0.05, '%.2f' % y, ha='center', va='bottom')
plt.show()

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

Num_iterations = 1000
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
Max_sharpe = Sim_frame.iloc[Sim_frame['Sharpe'].idxmax()]
Min_std = Sim_frame.iloc[Sim_frame['STD'].idxmin()]

plt.style.use('seaborn')
plt.title('Efficient Frontier', size=20)
sc = plt.scatter(Sim_frame.STD,Sim_frame.Return,c=Sim_frame.Sharpe,cmap='RdYlGn', edgecolors='black')
plt.scatter(Max_sharpe[1], Max_sharpe[0], color='r', s=60)
plt.scatter(Min_std[1], Min_std[0], color='b', s=60)
plt.xlabel('Standard Deviation(%)', size=15)
plt.ylabel('Returns(%)', size=15)
#plt.xlim(3.0, 5.5)
#plt.ylim(0.70, 1.1)
cb = plt.colorbar(sc)
cb.ax.get_yaxis().labelpad = 20
cb.set_label('Sharpe Ratio', rotation=270)
#plt.savefig('Simulation_3.jpeg', dpi=800,quality=100)
plt.show()
