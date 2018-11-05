# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:12:02 2018

@author: ChanJung Kim
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import date
from scipy import stats

# Task 3
DJI = pd.read_csv('^DJI.csv')
GSPC = pd.read_csv('^GSPC.csv')

# Extract close price data 
DJI_Price = np.array(DJI['Close'])
GSPC_Price = np.array(GSPC['Close'])

# Denominator = previous year's return, Nominator = this year's return
DJI_Denominator = DJI_Price[:-1]
DJI_Nominator = DJI_Price[1:]
GSPC_Denominator = GSPC_Price[:-1]
GSPC_Nominator = GSPC_Price[1:]

# Log returns
DJI_Logreturn = np.log(DJI_Nominator/DJI_Denominator)
GSPC_Logreturn = np.log(GSPC_Nominator/GSPC_Denominator)

# Linear Regression
GSPC_Logreturn = sm.add_constant(GSPC_Logreturn)
Regression = sm.OLS(DJI_Logreturn, GSPC_Logreturn)
Result = Regression.fit()

# Critical t-value
Critical_Value = stats.t.ppf(0.975, len(DJI_Logreturn)-2)

# Jarque-Bera test
Gamma = np.sum((Result.resid - Result.resid.mean())**3/len(Result.resid)*Result.resid.std()**3)
Kappa = np.sum((Result.resid - Result.resid.mean())**4/len(Result.resid)*Result.resid.std()**4)
JB = len(Result.resid)*((Gamma**2)/6 + ((Kappa-3)**2)/24)

# Task 4

# Extract the last working day of each year
Start = date(1985, 1, 29)
End = date(2018, 10, 17)
Last_Monthly_BD = pd.date_range(Start, End, freq='BM')

DJI['Date'] = pd.to_datetime(DJI['Date'])
GSPC['Date'] = pd.to_datetime(GSPC['Date'])
DJI.index = DJI['Date']
GSPC.index = GSPC['Date']

Last_Annual_BD =[]

if Last_Annual_BD == []:
    for i in Last_Monthly_BD:
        if i.month == 12:
            Last_Annual_BD.append(i)

# Extract log return for each index
DJI_Annual_Price = np.array(DJI['Close'].loc[Last_Annual_BD])
GSPC_Annual_Price = np.array(GSPC['Close'].loc[Last_Annual_BD])

# Denominator = previous year's return, Nominator = this year's return
DJI_Annual_Denominator = DJI_Annual_Price[:-1]
DJI_Annual_Nominator = DJI_Annual_Price[1:]
GSPC_Annual_Denominator = GSPC_Annual_Price[:-1]
GSPC_Annual_Nominator = GSPC_Annual_Price[1:]

# Log returns
DJI_Annual_Logreturn = np.log(DJI_Annual_Nominator/DJI_Annual_Denominator)
GSPC_Annual_Logreturn = np.log(GSPC_Annual_Nominator/GSPC_Annual_Denominator)

# Linear Regression for Annual Return
GSPC_Annual_Logreturn = sm.add_constant(GSPC_Annual_Logreturn)
Annual_Regression = sm.OLS(DJI_Annual_Logreturn, GSPC_Annual_Logreturn)
Annual_Result = Annual_Regression.fit()

# Critical t-value
Annual_Critical_Value = stats.t.ppf(0.975, len(DJI_Annual_Logreturn)-2)

# Jarque-Bera test for Annual Return
Annual_Gamma = np.sum((Annual_Result.resid - Annual_Result.resid.mean())**3/len(Annual_Result.resid)*Annual_Result.resid.std()**3)
Annual_Kappa = np.sum((Annual_Result.resid - Annual_Result.resid.mean())**4/len(Annual_Result.resid)*Annual_Result.resid.std()**4)
Annual_JB = len(Annual_Result.resid)*((Annual_Gamma**2)/6 + ((Annual_Kappa-3)**2)/24)

print("Task3-------------------------------------------------------------------------")
print("1. Alpha = %0.6f, Beta = %0.6f" %(Result.params[0], Result.params[1]))
print("2. STD of Residual = %0.6f" %(Result.resid.std()))
print("3. T-stats for a hat = %0.6f, T-stats for b hat = %0.6f" %(Result.tvalues[0], Result.tvalues[1]))
print("4. Critical value for 5%% significance level = %0.4f, %0.4f" %(float(-Critical_Value), float(Critical_Value)))
print("5. R-Squared = %0.6f, adjusted R-Squared = %0.6f" %(Result.rsquared, Result.rsquared_adj))
print("6. Jarque-Bera test stats = %0.6f" %(float(JB)))


print("Task4-------------------------------------------------------------------------")
print("1. Alpha = %0.6f, Beta = %0.6f" %(Annual_Result.params[0], Annual_Result.params[1]))
print("2. STD of Residual = %0.6f" %(Annual_Result.resid.std()))
print("3. T-stats for a hat = %0.6f, T-stats for b hat = %0.6f" %(Annual_Result.tvalues[0], Annual_Result.tvalues[1]))
print("4. Critical value for 5%% significance level = %0.4f, %0.4f" %(float(-Annual_Critical_Value), float(Annual_Critical_Value)))
print("5. R-Squared = %0.6f, adjusted R-Squared = %0.6f" %(Annual_Result.rsquared, Annual_Result.rsquared_adj))
print("6. Jarque-Bera test stats = %0.6f" %(float(Annual_JB)))
