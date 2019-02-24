# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:53:50 2018

@author: ChanJung Kim
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy.linalg import pinv
from matplotlib import pyplot as plt
import scipy.optimize as solver

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

plt.style.use('seaborn-notebook')
plt.title('Sharpe Ratio', size=20, y=1.05)
plt.bar(Sharpe_R.index, Sharpe_R, facecolor='black', edgecolor='black')
plt.xlabel('Industry', size=15)
plt.ylabel('Sharpe Ratio', size=15)
plt.ylim(0.00, 0.30)
plt.xticks(Sharpe_R.index, Sharpe_R.index, color='black', rotation=0,  fontsize='10', horizontalalignment='center')
for x, y in zip(Sharpe_R.index, Sharpe_R):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x, y+0.02,'%.2f' %y, ha='center', va='top')
plt.savefig('Sharpe_Ratio.jpeg', dpi=800, quality=100)
plt.show()

# Sortino Ratio
Sortino_RI_RF = RI_RF.copy()
Sortino_RI_RF[Sortino_RI_RF>0] =0
SV = (np.mean(np.square(Sortino_RI_RF)))*(len(Sortino_RI_RF)/(len(Sortino_RI_RF)-1))
Sortino_R = RI_RF.mean()/np.sqrt(SV)
print("2. Sortino Ratio for each industry")
print(np.sqrt(SV))
print(Sortino_R)

plt.title('Sortino Ratio', size=20, y=1.05)
plt.bar(Sortino_R.index, Sortino_R, facecolor='black', edgecolor='black')
plt.xlabel('Industry', size=15)
plt.ylabel('Sortino Ratio', size=15)
plt.ylim(0.00, 0.4)
plt.xticks(Sortino_R.index, Sortino_R.index, color='black', rotation=0,  fontsize='10', horizontalalignment='center')
for x, y in zip(Sortino_R.index, Sortino_R):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x , y+0.03, '%.2f' % y, ha='center', va='top')
plt.savefig('Sortino_Ratio.jpeg', dpi=800, quality=100)
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

plt.title('Jensen\'s Alpha', size=20, y=1.05)
plt.bar(Jensen_Alpha.index, Jensen_Alpha, facecolor='black', edgecolor='black')
plt.xlabel('Industry', size=15)
plt.ylabel('Jensen\'s Alpha', size=15)
plt.ylim(-0.5, 0.65)
plt.xticks(Jensen_Alpha.index, Jensen_Alpha.index, color='black', rotation=0,  fontsize='10', horizontalalignment='center')
for x, y in zip(Jensen_Alpha.index, Jensen_Alpha):
    if y >= 0:
        plt.text(x , y+0.07, '%.2f' % y, ha='center', va='top')
    elif y < 0:
        plt.text(x , y-0.07, '%.2f' % y, ha='center', va='bottom')
plt.savefig('Jensen_Alpha.jpeg', dpi=800,quality=100)
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

plt.title('Three-Factor Alpha', size=20, y=1.05)
plt.bar(Three_Factor_Alpha.index, Three_Factor_Alpha, facecolor='black', edgecolor='black')
plt.xlabel('Industry', size=15)
plt.ylabel('Three-Factor Alpha', size=15)
plt.ylim(-0.6, 0.65)
plt.xticks(Three_Factor_Alpha.index, Three_Factor_Alpha.index, color='black', rotation=0,  fontsize='10', horizontalalignment='center')
for x, y in zip(Three_Factor_Alpha.index, Three_Factor_Alpha):
    if y >= 0:
        plt.text(x , y+0.08, '%.2f' % y, ha='center', va='top')
    elif y < 0:
        plt.text(x , y-0.08, '%.2f' % y, ha='center', va='bottom')
plt.savefig('Three_Factor_Alpha.jpeg', dpi=800,quality=100)
plt.show()


# Part2 Minumum Variance
# Simulation
Num_iterations = 100_000
Simulation_res = np.zeros((3, Num_iterations))

for i in range(Num_iterations):
    Weights = 1/np.random.random(len(IP.columns))
    Weights /= np.sum(Weights)
    Portfolio_return = np.sum(IP.mean() * Weights)
    Portfolio_std_dev = np.sqrt(np.dot(Weights.T,np.dot(IP.cov(), Weights)))
    Simulation_res[0,i] = Portfolio_return
    Simulation_res[1,i] = Portfolio_std_dev
    Simulation_res[2,i] = Simulation_res[0,i] / Simulation_res[1,i]

Sim_frame = pd.DataFrame(Simulation_res.T, columns = ['Return', 'STD', 'Sharpe'])
Max_sharpe = Sim_frame.iloc[Sim_frame['Sharpe'].idxmax()]
Min_std = Sim_frame.iloc[Sim_frame['STD'].idxmin()]

# Efficient Frontier
def sd(w):
    return np.sqrt(np.dot(w, np.dot(IP.cov(), w.T)))

x0=np.array([1.0/10 for x in range(10)])
bounds=tuple((0,1) for x in range(10))

given_r=np.arange(0.50,1.20,0.001)
risk=[]
for i in given_r:
    constraints=[{'type':'eq','fun':lambda x: sum(x)-1},
                 {'type':'eq','fun':lambda x: sum(x*IP.mean())-i}]
    outcome=solver.minimize(sd,x0=x0,constraints=constraints,bounds=bounds)
    risk.append(outcome.fun)

# Plug-in Efficient frontier from Assignment1

V = np.matrix(IP.cov())
R = np.matrix(IP.mean())
R = R.reshape(R.size, 1)

I = np.ones((len(R),1))
VAR = np.diag(V)
SD = np.sqrt(np.diag(V))

Delta = I.T*pinv(V)*I 
Alpha = R.T*pinv(V)*I
Zeta = R.T*pinv(V)*R
D = Zeta*Delta-Alpha**2

mu = np.arange(0,2,2/10000)
minvar = (Zeta-2*Alpha*mu+(Delta*mu**2))/D;
minstd = np.sqrt(minvar)[0]
minstd = np.squeeze(np.asarray(minstd))

plt.figure(figsize=(15,8))
plt.style.use('seaborn-notebook')
plt.title('Efficient Frontier', size=20)
sc = plt.scatter(Sim_frame.STD,Sim_frame.Return,c=Sim_frame.Sharpe,cmap='RdYlGn', edgecolors='black', label ='Simulation Result')
#sc = plt.scatter(Sim_frame.STD,Sim_frame.Return,c=Sim_frame.Sharpe,cmap='Greys',s=4)
plt.plot(minstd, mu, color='k', label='Efficient Frontier without portfolio weights restriction', linewidth=1.0, linestyle='--')
plt.plot(risk,given_r,'--',color='b', label='Efficient Frontier Using Optimization')
#plt.scatter(Max_sharpe[1], Max_sharpe[0], color='r', s=60, label='Maximum Sharpe Ratio')
#plt.scatter(Min_std[1], Min_std[0], color='b', s=60, label='Minimum Variance')
plt.xlabel('Standard Deviation(%)', size=15)
plt.ylabel('Returns(%)', size=15)
cb = plt.colorbar(sc)
cb.ax.get_yaxis().labelpad = 20
cb.set_label('Sharpe Ratio', rotation=270)
plt.legend(loc='upper center', frameon=True, fancybox=True)
plt.grid(True)
plt.show()
