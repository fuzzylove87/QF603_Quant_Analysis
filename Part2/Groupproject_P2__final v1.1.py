# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:29:41 2018
@author: Group F
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statistics

#Plugging in previous definitions created in Assignment 1
def Skewness(ret_list):
    mean_ret=np.mean(ret_list)
    std_ret=np.std(ret_list,ddof=0)
    skew_num_list=[((i-(mean_ret))**3) for i in ret_list]
    return np.sum(skew_num_list)/(len(ret_list)*(std_ret**3))

def Kurtosis(ret_list):
    mean_ret=np.mean(ret_list)
    std_ret=np.std(ret_list,ddof=0)
    kurt_num_list=[((i-(mean_ret))**4) for i in ret_list]
    return np.sum(kurt_num_list)/(len(ret_list)*(std_ret**4))

def Jarque_Beta(ret_list):
    JB = len(ret_list)*((Skewness(ret_list)**2)/6 + ((Kurtosis(ret_list)-3)**2)/24)
    return JB

print("---------------------------- Task 3 ----------------------------")

DJI = pd.read_csv('DJI.csv', index_col=0,parse_dates=['Date'])
GSPC = pd.read_csv('GSPC.csv',index_col=0,parse_dates=['Date'])

lreturns_DJI_y = (np.log(DJI.Close)-np.log(DJI.Close.shift(1)))[1:]
lreturns_GSPC_x = (np.log(GSPC.Close)-np.log(GSPC.Close.shift(1)))[1:]

lreturns_GSPC_x_mean = np.mean(lreturns_GSPC_x)
lreturns_DJI_y_mean = np.mean(lreturns_DJI_y)

xd = lreturns_GSPC_x - lreturns_GSPC_x_mean
yd = lreturns_DJI_y - lreturns_DJI_y_mean

n = len(xd)

# 1. ahat and bhat
bhat = sum(xd*yd)/sum(xd*xd)
ahat = lreturns_DJI_y_mean - bhat * lreturns_GSPC_x_mean
print("1. ahat = %0.5f bhat = %0.5f" % (ahat, bhat))

# fitted value yf and residual e
yf = ahat + bhat * lreturns_GSPC_x
e = lreturns_DJI_y - yf

# Total sum of squares and residual sum of squares
tss = sum(yd**2)
rss = sum(e**2)

# 2 .sigma of residuals
sigma_e2 = rss/(n-2)
sigma = (rss/(n-2))**0.5
print("2. sigma = %0.5f" % sigma)

# standard errors
xss = sum(xd**2)
SE_bhat = np.sqrt(sigma_e2/xss)
SE_ahat = np.sqrt(sigma_e2 * (1/n + (lreturns_GSPC_x_mean**2)/xss))

# 3. t statistics
tstat0 = ahat/SE_ahat
tstat1 = bhat/SE_bhat
print("3. tstat of ahat = %0.5f and tstat of bhat = %0.5f" % (tstat0, tstat1))

# 4. critical value for t statistics
t_critical = stats.t.ppf(1-0.025, n-2)
print("4. critical value for two-tail t test =  +%0.5f , -%0.5f" % (t_critical,t_critical))

# 5. R square
R2 = 1 - rss/tss
print("5. R^2 = %0.5f percent" % (R2*100))

# 5. Adjusted R square
R2bar = 1 - ((rss/(n-2))/(tss/(n-1)))
print("5. Adjusted R^2 = %0.5f percent" % (R2bar*100))

#6. Jarque-Bera test statistics
JB_residual = Jarque_Beta(e)
Chisquare_value = stats.chi2.ppf(1-0.05,2)
print("6. Jarque-Bera test stats = %0.5f" % (JB_residual))
print("7. Chi-square critical value forJarque-Bera test = %0.5f" % (Chisquare_value))

#8 Additional findings
tstat2 = (bhat-1)/SE_bhat
print("8. tstat of bhat (when H0: b=1) = %0.5f" % (tstat2))

x_plots=[np.min(lreturns_GSPC_x),np.max(lreturns_GSPC_x)]
y_plots=[(ahat + bhat*i) for i in x_plots]

plt.plot(x_plots,y_plots,label='Regression Line',color='k')
plt.scatter(lreturns_GSPC_x,lreturns_DJI_y,label='Scatter Plot of S&P 500 vs DJIA Daily Log Returns',color='r')
plt.xlabel('S&P 500 Daily Log Returns')
plt.ylabel('DJIA Daily Log Returns')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()

plt.hist([i*100 for i in e],bins=50,density=True,label='Density Histogram of Residual of Daily Log Returns',color='r')
plt.xlabel('Residuals (%)')
plt.ylabel('Density')
plt.legend(loc='upper left')    
plt.grid(True)
plt.axis([np.min(e_annual*100)*1.2,np.max(e_annual*100)*1.2,0,2.2])
plt.show()

print("---------------------------- Task 4 ----------------------------")

DJI_Close_annual = DJI[DJI.index.month.isin([12])].groupby(DJI[DJI.index.month.isin([12])].index.year).last()
GSPC_Close_annual = GSPC[GSPC.index.month.isin([12])].groupby(GSPC[GSPC.index.month.isin([12])].index.year).last()

lreturns_DJI_y_annual = (np.log(DJI_Close_annual.Close)-np.log(DJI_Close_annual.Close.shift(1)))[1:]
lreturns_GSPC_x_annual = (np.log(GSPC_Close_annual.Close)-np.log(GSPC_Close_annual.Close.shift(1)))[1:]

lreturns_GSPC_x_mean_annual = np.mean(lreturns_GSPC_x_annual)
lreturns_DJI_y_mean_annual = np.mean(lreturns_DJI_y_annual)

xd_annual = lreturns_GSPC_x_annual - lreturns_GSPC_x_mean_annual
yd_annual = lreturns_DJI_y_annual - lreturns_DJI_y_mean_annual

n_annual = len(xd_annual)

# 1. ahat and bhat
bhat_annual = sum(xd_annual*yd_annual)/sum(xd_annual*xd_annual)
ahat_annual = lreturns_DJI_y_mean_annual - bhat_annual * lreturns_GSPC_x_mean_annual
print("1. ahat = %0.5f bhat = %0.5f" % (ahat_annual, bhat_annual))

# fitted value yf and residual e
yf_annual = ahat_annual + bhat_annual * lreturns_GSPC_x_annual
e_annual = lreturns_DJI_y_annual - yf_annual

# Total sum of squares and residual sum of squares
tss_annual = sum(yd_annual**2)
rss_annual = sum(e_annual**2)

# 2 .sigma of residuals
sigma_e2_annual = rss_annual/(n_annual-2)
sigma_annual = (rss_annual/(n_annual-2))**0.5
print("2. sigma = %0.5f" % sigma_annual)

# standard errors
xss_annual = sum(xd_annual**2)
SE_bhat_annual = np.sqrt(sigma_e2_annual/xss_annual)
SE_ahat_annual = np.sqrt(sigma_e2_annual * (1/n_annual + (lreturns_GSPC_x_mean_annual**2)/xss_annual))

# 3. t statistics
tstat0_annual = ahat_annual/SE_ahat_annual
tstat1_annual = bhat_annual/SE_bhat_annual
print("3. tstat of ahat = %0.5f and tstat of bhat = %0.5f" % (tstat0_annual, tstat1_annual))

# 4. critical value for t statistics
t_critical_annual = stats.t.ppf(1-0.025, n_annual-2)
print("4. critical value for two-tail t test =  +%0.5f , -%0.5f" % (t_critical_annual,t_critical_annual))

# 5. R square
R2_annual = 1 - rss_annual/tss_annual
print("5. R^2 = %0.5f percent" % (R2_annual*100))

# 5. Adjusted R square
R2bar_annual = 1 - ((rss_annual/(n_annual-2))/(tss_annual/(n_annual-1)))
print("5. Adjusted R^2 = %0.5f percent" % (R2bar_annual*100))

#6. Jarque-Bera test statistics
JB_residual_annual = Jarque_Beta(e_annual)
print("6. Jarque-Bera test stats = %0.5f" % (JB_residual_annual))
print("7. Chi-square critical value for Jarque-Bera test = %0.5f" % (Chisquare_value))

#8 Additional findings
tstat2_annual = (bhat_annual-1)/SE_bhat_annual
print("8. tstat of bhat (when H0: b=1) = %0.5f" % (tstat2_annual))


x_plots_annual=[np.min(lreturns_GSPC_x_annual),np.max(lreturns_GSPC_x_annual)]
y_plots_annual=[(ahat_annual + bhat_annual*i) for i in x_plots_annual]

plt.plot(x_plots_annual,y_plots_annual,label='Regression Line',color='k')
plt.scatter(lreturns_GSPC_x_annual,lreturns_DJI_y_annual,label='Scatter Plot of S&P 500 vs DJIA Annual Log Returns',color='b')
plt.xlabel('S&P 500 Annual Log Returns')
plt.ylabel('DJIA Annual Log Returns')
plt.grid(True)
plt.legend(loc='lower right')
plt.show()


plt.hist([i*100 for i in e_annual],bins=4,density=True,label='Density Histogram of Residuals of Annual Log Returns',color='b')
plt.xlabel('Residuals (%)')
plt.ylabel('Density')
plt.legend(loc='upper left')    
plt.grid(True)
plt.axis([np.min(e_annual*100)*1.2,np.max(e_annual*100)*1.2,0,2.2])
plt.show()
