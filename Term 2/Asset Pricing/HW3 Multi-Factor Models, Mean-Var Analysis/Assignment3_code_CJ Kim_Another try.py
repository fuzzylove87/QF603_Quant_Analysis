# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:00:28 2018

@author: ChanJung Kim
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
from matplotlib import pyplot as plt

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
VAR_RIRT = np.mean(np.square(RI_RF - RI_RF.mean()))
Sortino_R = RI_RF.mean()/np.sqrt(VAR_RIRT)
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

# Part2
