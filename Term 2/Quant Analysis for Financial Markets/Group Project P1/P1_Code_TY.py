# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:42:13 2018

@author: Woon Tian Yong
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
'''
# Task 1.1

DJI_Comp=pd.ExcelFile('DJIA Price Data.xlsx')
DJI_Comp_Prices=(DJI_Comp.parse('Prices'))['Price']

divisor=0.14748071991788

DJI_price_computed=(DJI_Comp_Prices.sum()/divisor)

print('------------------------------------------------------------------------------')  
print('Computed Dow Jones Ind Avg Price: '+str(round(DJI_price_computed,4)))
print('------------------------------------------------------------------------------')  
'''
# Task 1.3

DJI=pd.read_csv('DJI.csv')
GSPC=pd.read_csv('GSPC.csv')

GSPC_Prices=list(GSPC['Close'])
DJI_Prices=list(DJI['Close'])

Dates=[]
for i in list(DJI['Date']):
    Dates.append(datetime.strptime(i,'%Y-%m-%d'))
    
Chart, DJI_Plot=plt.subplots()

DJI_Plot.plot(Dates,DJI_Prices,label='Dow Jones Industrial Average',color='b')
DJI_Plot.set_xlabel('Year')
DJI_Plot.set_ylabel('Dow Jones Ind Avg Price')

GSPC_Plot=DJI_Plot.twinx()

GSPC_Plot.plot(Dates,GSPC_Prices,label='S&P 500',color='r')
GSPC_Plot.set_ylabel('S&P 500 Price')

Chart.tight_layout()
Chart.legend(loc=(0.16,0.82))
plt.savefig('time_series.png', dpi=900)
plt.show()


# Task 1.4

DJI_LogReturns=[np.log(DJI_Prices[i+1]/DJI_Prices[i]) for i in range(len(DJI_Prices)-1)]
    
GSPC_LogReturns=[np.log(GSPC_Prices[i+1]/GSPC_Prices[i]) for i in range(len(GSPC_Prices)-1)]

t_list=[i+1 for i in range(len(DJI_LogReturns))]

# Task 1.5

plt.plot(t_list,[i*100 for i in DJI_LogReturns],label='Dow Jones Ind Avg Log Returns',color='b')
plt.xlabel('Period')
plt.ylabel('Log Return (%)')
plt.axis([0,8500,-28,28])
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig('DJI_logret.png')
plt.show()

plt.plot(t_list,[i*100 for i in GSPC_LogReturns],label='S&P 500 Log Returns',color='r')
plt.xlabel('Period')
plt.ylabel('Log Return (%)')
plt.axis([0,8500,-28,28])
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig('SP_logret.png')
plt.show()


# Task 1.6

DJI_Mean=np.mean(DJI_LogReturns)
GSPC_Mean=np.mean(GSPC_LogReturns)

DJI_Var=np.var(DJI_LogReturns,ddof=1)
GSPC_Var=np.var(GSPC_LogReturns,ddof=1)

print('------------------------------------------------------------------------------')  
print('Dow Jones Ind Avg Mean Daily Log Return: ' + str(round(DJI_Mean,8)))
print('S&P 500 Mean Log Daily Return: ' + str(round(GSPC_Mean,8)))
print('------------------------------------------------------------------------------')  
print('Dow Jones Ind Avg Sample Variance of Returns: ' + str(round(DJI_Var,8)))
print('S&P 500 Sample Variance of Returns: ' + str(round(GSPC_Var,8)))

# Task 1.7

DJI_MeanAnn=DJI_Mean*252
GSPC_MeanAnn=GSPC_Mean*252

DJI_VarAnn=(DJI_Var*252)**0.5
GSPC_VarAnn=(GSPC_Var*252)**0.5

print('------------------------------------------------------------------------------')  
print('Dow Jones Ind Avg Annualised Return: ' + str(round(DJI_MeanAnn,8)))
print('S&P 500 Annualised Return: ' + str(round(GSPC_MeanAnn,8)))
print('------------------------------------------------------------------------------')  
print('Dow Jones Ind Avg Annualised Volatility: ' + str(round(DJI_VarAnn,8)))
print('S&P 500 Annualised Volatility: ' + str(round(GSPC_VarAnn,8)))

# Task 1.8

def Skewedness(ret_list):
    mean_ret=np.mean(ret_list)
    std_ret=np.std(ret_list,ddof=0)
    skew_num_list=[((i-(mean_ret**2))**3) for i in ret_list]
    return np.sum(skew_num_list)/(len(ret_list)*(std_ret**3))


def Kurtosis(ret_list):
    mean_ret=np.mean(ret_list)
    std_ret=np.std(ret_list,ddof=0)
    kurt_num_list=[((i-(mean_ret**2))**4) for i in ret_list]
    return np.sum(kurt_num_list)/(len(ret_list)*(std_ret**4))

DJI_Skew=Skewedness(DJI_LogReturns)
GSPC_Skew=Skewedness(GSPC_LogReturns)

DJI_Kurt=Kurtosis(DJI_LogReturns)
GSPC_Kurt=Kurtosis(GSPC_LogReturns)

print('------------------------------------------------------------------------------')  
print('Skewedness of Dow Jones Ind Avg\'s Returns: ' + str(round(DJI_Skew,8)))
print('Skewedness of S&P 500\'s Returns: ' + str(round(GSPC_Skew,8)))
print('------------------------------------------------------------------------------')  
print('Kurtosis of Dow Jones Ind Avg\'s Returns: ' + str(round(DJI_Kurt,8)))
print('Kurtosis of S&P 500\'s Returns: ' + str(round(GSPC_Kurt,8)))

# Task 1.9

def Jarque_Beta(ret_list):
    n=len(ret_list)
    return n*(((Skewedness(ret_list)**2)/6)+(((Kurtosis(ret_list)-3)**2)/24))

DJI_JBStat=Jarque_Beta(DJI_LogReturns)
GSPC_JBStat=Jarque_Beta(GSPC_LogReturns)

print('------------------------------------------------------------------------------')  
print('Jarque-Beta Statistic of Dow Jones Ind Avg\'s Returns: '+str(round(DJI_JBStat,2)))
print('Jarque-Beta Statistic of S&P 500\'s Returns: '+str(round(GSPC_JBStat,2)))
print('')
print('Critical Chi-Square Value with 2 Degrees of Freedom: '+ str(round(stats.chi2.ppf(1-0.05,2),6)))
print('')
print('Thus, able to reject null hypothesis that JB=0 for both indices.')

# Task 2.1

DJI_GSPC_Corr=np.corrcoef(DJI_LogReturns,GSPC_LogReturns)[1,0]
print('------------------------------------------------------------------------------')  
print('Correlation: '+str(round(DJI_GSPC_Corr,6)))

# Task 2.2

def tStat_MeanDiff(ret_list1,ret_list2):
    mean_ret1=np.mean(ret_list1)
    mean_ret2=np.mean(ret_list2)
    var_ret1=np.var(ret_list1,ddof=1)
    var_ret2=np.var(ret_list2,ddof=1)
    n1=len(ret_list1)
    n2=len(ret_list2)
    return ((mean_ret1-mean_ret2)/
            (np.sqrt(var_ret1/n1+var_ret2/n2)))
    
def tStat_MeanDiff_DF(ret_list1,ret_list2):
    var_ret1=np.var(ret_list1,ddof=1)
    var_ret2=np.var(ret_list2,ddof=1)
    n1=len(ret_list1)
    n2=len(ret_list2)
    return (((var_ret1/n1+var_ret2/n2)**2)/
            ((((var_ret1/n1)**2)/(n1-1))+((var_ret2/n2)**2)/(n2-1)))
    
DJI_GPSC_MeanDiff_tStat=tStat_MeanDiff(DJI_LogReturns,GSPC_LogReturns)
DJI_GPSC_MeanDiff_DF=tStat_MeanDiff_DF(DJI_LogReturns,GSPC_LogReturns)
    
print('------------------------------------------------------------------------------')     
print('t Statistic: '+str(round(DJI_GPSC_MeanDiff_tStat,6))) 
print('')
print('Critical t Values: +/-'+ str(round(stats.t.ppf(1-0.025,DJI_GPSC_MeanDiff_DF),6)))
print('')
print('Thus, unable to reject null hypothesis that mean differences are 0.')

# Task 1.3

def F_Stat(ret_list1,ret_list2):
    var_ret1=np.var(ret_list1,ddof=1)
    var_ret2=np.var(ret_list2,ddof=1)
    return max(var_ret1/var_ret2,var_ret2/var_ret1)

def F_Stat_DFs(ret_list1,ret_list2):
    var_ret1=np.var(ret_list1,ddof=1)
    var_ret2=np.var(ret_list2,ddof=1)
    n1=len(ret_list1)
    n2=len(ret_list2)
    if var_ret1>var_ret2:
        return [n1-1,n2-1]
    elif var_ret1<var_ret2:
        return [n2-1,n1-1]
    else:
        print('ERROR')

DJI_GPSC_FStat=F_Stat(DJI_LogReturns,GSPC_LogReturns)
DJI_GPSC_FStat_DFs=F_Stat_DFs(DJI_LogReturns,GSPC_LogReturns)

print('------------------------------------------------------------------------------')   
print('F Statistic: '+str(round(DJI_GPSC_FStat,6))) 
print('')
print('Critical F Values: '+ str(round(stats.f.ppf(0.05/2,DJI_GPSC_FStat_DFs[0],DJI_GPSC_FStat_DFs[1]),6))+', '+
                             str(round(stats.f.ppf(1-0.05/2,DJI_GPSC_FStat_DFs[0],DJI_GPSC_FStat_DFs[1]),6)))
print('')
print('Thus, able to reject null hypothesis that the 2 population variances are equal.')
print('------------------------------------------------------------------------------')  
