"""
2018-10-25
Mini project Part 1 Scientific Correctness
Christopher Ting
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, t, f

DJIA = pd.read_csv('^DJI.csv')
DJIA.index = pd.to_datetime(DJIA.Date)
# plt.plot(DJIA.Close)

# More profesional plot but can be even more professional
SP500 = pd.read_csv('^GSPC.csv')
SP500.index = pd.to_datetime(SP500.Date)
# plt.plot(SP500.Close)

LD, LS = np.log(DJIA.Close), np.log(SP500.Close)
rD, rS = np.diff(LD), np.diff(LS)
lr = pd.DataFrame(data = {'DJIA': rD, 'SP500': rS})
lr.index = DJIA.index[1:]
# plt.plot(lr.DJIA)
# plt.show()

a = lr.mean()
s = lr.std(ddof=1)

ydays = 252

A, V = a*ydays,  s*np.sqrt(ydays)

print('Annualized Log Return on %s = %0.2f%%' % (A.index[0], A.iloc[0]*100))
print('Annualized Log Return on %s = %0.2f%%' % (A.index[1], A.iloc[1]*100))
print('Annualized Volatility of %s = %0.2f%%' % (V.index[0], V.iloc[0]*100))
print('Annualized Volatility of %s = %0.2f%%' % (V.index[1], V.iloc[1]*100))

lr_demeaned = lr - a

T, alpha = len(lr), 0.05

sigma = s*np.sqrt((T-1)/T)

skewness0 = sum(lr_demeaned.DJIA**3)/(T* sigma.DJIA**3)
skewness1 = sum(lr_demeaned.SP500**3)/(T* sigma.SP500**3)

kurtosis0 = sum(lr_demeaned.DJIA**4)/(T* sigma.DJIA**4) 
kurtosis1 = sum(lr_demeaned.SP500**4)/(T* sigma.SP500**4)  

JB0 = T*(skewness0**2/6 + (kurtosis0-3)**2/24)
JB1 = T*(skewness1**2/6 + (kurtosis1-3)**2/24) 
cvalue = chi2.ppf(1-alpha, 2)
print('\nJB of DJIA = %0.0f, JB of SP500 = %0.0f, Critical Value = %0.2f' 
     %  (JB0, JB1,  cvalue))
print('The null hypothesis of normality must be rejected.')

###############################################################################
correlation = np.corrcoef(lr.DJIA, lr.SP500)
print('\nCorrelation between DJIA and S&P500 log returns = %0.2f%%' 
      %  (correlation[0][1]*100))

T_test_stat = (a.DJIA - a.SP500)/np.sqrt(s.DJIA**2/T + s.SP500**2/T )
sT0, sT1 = s.DJIA**2/T, s.SP500**2/T

ddof = (sT0 + sT1)**2/(sT0**2/(T-1) + sT1**2/(T-1))
t_cvalue = t.ppf(1-alpha/2, ddof)
print('\nTwo sample t test stat = %0.4f, Critical Value = %0.2f'
      % (T_test_stat, t_cvalue))
print('The null hypothesis of same mean cannot be rejected.')

F = s.DJIA**2/s.SP500**2

f_cvalue_lower = f.ppf(alpha/2, T-1, T-1)
f_cvalue_upper = f.ppf(1-alpha/2, T-1, T-1)

print('\nF test stat = %0.4f, Lower crit val = %0.4f, Upper crit val = %0.4f'
      % (F, f_cvalue_lower, f_cvalue_upper))
print('The null hypothesis of equal variance must be rejected.')






