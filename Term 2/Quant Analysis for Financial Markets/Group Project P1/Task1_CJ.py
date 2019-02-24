# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:29:41 2018

@author: ykfri
"""

from matplotlib import pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from scipy import stats
import csv
import math

# Task 1
Oct18 = pd.read_csv('DJIA_components.csv')
TotalPrice = Oct18[' Price:D-1 '].sum()
divisor = 0.14748071991788
print(TotalPrice)
print(Oct18.iloc[:, 0:3])
print(TotalPrice/divisor)

# Task 3
DJI = pd.read_csv('DJI.csv')
GSPC = pd.read_csv('GSPC.csv')

DJI_close = DJI['Close']
GSPC_close = GSPC['Close']

Dates = DJI['Date']
'''
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

DJI_line = ax1.plot(DJI_close, color='b', label='DJI')
GSPC_line = ax2.plot(GSPC_close, color='r', label='GSPC')

Line = DJI_line + GSPC_line
labs = [l.get_label() for l in Line]
plt.rc('grid', linestyle="--", color='black')
ax1.legend(Line, labs, loc='lower right', facecolor='white', framealpha=2)
plt.title('Time series of DJI and GSPC',fontsize=20)
plt.grid()
plt.show()
'''
Chart, DJI_Plot=plt.subplots()

DJI_Plot.plot(Dates,DJI_close,label='Dow Jones Industrial Average',color='b')
DJI_Plot.set_xlabel('Year')
DJI_Plot.set_ylabel('Dow Jones Ind Avg Price')

GSPC_Plot=DJI_Plot.twinx()

GSPC_Plot.plot(Dates,GSPC_close,label='S&P 500',color='r')
GSPC_Plot.set_ylabel('S&P 500 Price')

Chart.tight_layout()
Chart.legend(loc=(0.16,0.82))
plt.show()




# Task 4, 5
DJI_close = np.array(DJI_close)
DJI_close1 = DJI_close[:-1]
DJI_close2 = DJI_close[1:]

DJIret = DJI_close2/DJI_close1
DJIlogret = np.log(DJIret)

plt.title('Log Return of Dow Jones Index',fontsize=20)
plt.plot(DJIlogret)
plt.show()

GSPC_close = np.array(GSPC_close)
GSPC_close1 = GSPC_close[:-1]
GSPC_close2 = GSPC_close[1:]

GSPCret = GSPC_close2/GSPC_close1
GSPClogret = np.log(GSPCret)

plt.title('Log Return of S&P500',fontsize=20)
plt.plot(GSPClogret)
plt.show()

# Task 6
DJImean = DJIlogret.mean()
GSPCmean = GSPClogret.mean()
print('Sample mean of DJI is ' +str(DJImean))
print('Sample mean of S&P 500 is ' +str(GSPCmean))


DJIvar = np.var(DJIlogret, ddof=1)
GSPCvar = np.var(GSPClogret, ddof=1)
print('Sample variance of DJI is ' +str(DJIvar))
print('Sample variance of S&P 500 is ' +str(GSPCvar))

# Task 7
DJIAmean = 252*DJImean
GSPCAmean = 252*GSPCmean
print(DJIAmean)
print(GSPCAmean)

DJIAstd = math.sqrt(252*DJIvar)
GSPCAstd = math.sqrt(252*GSPCvar)
print(DJIAstd)
print(GSPCAstd)

# Task 8

def Skewedness(logreturn):
    Meanmatrix = np.ones(len(logreturn))*np.mean(logreturn)
    Parentheses = logreturn - Meanmatrix
    Withinsigma = np.power(Parentheses, 3)
    Nominator = np.sum(Withinsigma)

    Sigmasquare = np.sum(np.power((logreturn - Meanmatrix), 2))/len(logreturn)
    Denominator = len(logreturn)*math.sqrt(Sigmasquare)**3
    Gamma = Nominator/Denominator
    return Gamma

def Kurtosis(logreturn):
    Meanmatrix = np.ones(len(logreturn))*np.mean(logreturn)
    Parentheses = logreturn - Meanmatrix
    Withinsigma = np.power(Parentheses, 4)
    Nominator = np.sum(Withinsigma)
    
    Sigmasquare = np.sum(np.power((logreturn - Meanmatrix), 2))/len(logreturn)
    Denominator = len(logreturn)*math.sqrt(Sigmasquare)**4
    Kappa = Nominator/Denominator
    return Kappa

# Task 9
def JBstat(logreturn):
    JB = len(logreturn)*((Skewedness(logreturn)**2)/6 + ((Kurtosis(logreturn)-3)**2)/24)
    return JB

DJI_JBstat = JBstat(DJIlogret)
GSPC_JBstat = JBstat(GSPClogret)

print('Skewedness of DJI is '+ str(Skewedness(DJIlogret)))
print('Skewedness of SGPC is '+ str(Skewedness(GSPClogret)))

print('Kurtosis of DJI is '+ str(Kurtosis(DJIlogret)))
print('Kurtosis of SGPC is '+ str(Kurtosis(GSPClogret)))

print('Jarque-Bera test statistics of DJI is ' + str(JBstat(DJIlogret)))
print('Jarque-Bera test statistics of GSPC is ' + str(JBstat(GSPClogret)))

'''
Chisq_value = stats.chi2.ppf(1-0.05,2)



def Hypothesis_test(logreturn):
    if Chisq_value>JBstat(logreturn):
        print('Jarque-Beta Statistic of' + logreturn + 'Avg\'s Returns: '+str(round(JBstat(logreturn),2)))
        print('Critical Chi-Square Value with 2 Degrees of Freedom: '+ Chisq_value)
    else :
        print('Jarque-Beta Statistic of' + logreturn + 'Avg\'s Returns: '+str(round(JBstat(logreturn),2)))
        print('Critical Chi-Square Value with 2 Degrees of Freedom: '+ Chisq_value)
        
Hypothesis_test(DJIlogret)
'''




