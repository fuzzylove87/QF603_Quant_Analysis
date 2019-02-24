"""
source: http://www.temasekreview.com.sg/#overview-performanceOverview
2018-10-18 version 1
2018-10-18 version 2
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

"""" Capture the data """
# Portfolio value in billions of S$ as at 31 March
v = [90, 103, 129, 164, 185, 130, 186, 193]
y = range(2004, 2012)   # Perculatrity of range function,

performance = {'Portfolio_Value' : v}
df = pd.DataFrame.from_dict(performance)
df.index = y
print(df)

# Uncomment the following two lines to plot and show
# plt.plot(df, '-o')
# plt.show()

""" Compute simple returns """
ret = df.pct_change()   # Compute the simple return
ret = ret.rename(columns={'Portfolio_Value': 'Return'})
ret = ret.dropna()   # drop the first NaN

print('\nAnswer to Q1(a)')
i = 1
for a in ret.Return:
   print('%s\t%0.2f%%' % (y[i], a*100))
   i += 1

""" Test the hypothesis that mu = 0.07 """
print('\nAnswer to Q1(b)')
avg = ret.mean()
print('Average retrun = %0.2f%%' % (avg * 100))

print('\nAnswer to Q1(c)')
v = ret.var(ddof=1)   # unbiased with one degree of freedom 
vol = np.sqrt(v)
print('Volatility = %0.2f%%' % (vol * 100))

print('\nAnswer to Q1(d)')
n = len(ret)
se = np.sqrt(v/n)   # Standard error for in-sample hypothesis test
mu = 0.07
t = (avg - mu)/se
print('t statistic computed for testing hypothesis = %0.2f' % (t))
print('\nInference: Cannot reject the null.')
print('i.e., %0.2f%% is statistically no different from 7%%' % (avg * 100))

###############################################################################
""" Prediction by Model 0 """
print('\nUse Model 0 to predict out-of-sample')

s_f = vol * np.sqrt(1 + 1/n)   # standard deviation for prediction
print('\nAnswer to Q1(e)')
se_f = s_f/np.sqrt(n)          # standard error for prediction
print('Standard error of 2012 forceast = %0.2f%%' % (se_f*100))

t = stats.t.ppf(1-0.025, n-1)   # instead of lookig up table...

print('\nForcast by Model 0 = %0.2f%%' % (avg * 100))
print ('t stat with %d degrees of freedom = %0.2f' % (n-1, t))

print('\nAnswer to Q1(f)')
print('Upper bound of the forecast = %0.2f%%' % ((avg + se_f*t)*100))
print('Lower bound of the forecast = %0.2f%%' % ((avg - se_f*t)*100))

"""
It turn out that Temasek's portfolio was 198 billion in 2012,
hence the actual return is 2.59%,
which is within the upper and lower bounds
"""


