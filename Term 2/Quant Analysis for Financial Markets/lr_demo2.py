"""
Date: 2017-11-05, 2018-01-22
Author: Christopher Ting
Gold, Silver, and Iron
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# https://www.ptable.com/  
density = [19.30, 10.49, 7.874]  # g/cm^3

# word number of gold, silver, and iron
wn = [14, 160, 239]

ldensity = np.log(density)

x = np.asarray(wn)
y = ldensity

xa = np.mean(x)
ya = np.mean(y)

xd = x - xa
yd = y - ya

n = len(xd)

# estimates
bhat = sum(xd*yd)/sum(xd*xd)
ahat = ya - bhat*xa
print("ahat = %0.2f bhat = %0.4f" % (ahat, bhat))

# fitted value yf and residual e
yf = ahat + bhat * x
e = y - yf

# Total sum of squares and residual sum of squares
tss = sum(yd**2)
rss = sum(e**2)

# R square
R2 = 1 - rss/tss
print("R^2 = %0.2f percent" % (R2*100))

# variance of residuals
sigma_e2 = rss/(n-2)

# Adjusted R square
R2bar = 1 - sigma_e2/tss
print("Adjusted R^2 = %0.2f percent" % (R2bar*100))

# standard errors
xss = sum(xd**2)
SE_bhat = np.sqrt(sigma_e2/xss)
SE_ahat = np.sqrt(sigma_e2 * (1/n + (xa**2)/xss))

# t statistics
tstat0 = ahat/SE_ahat
tstat1 = bhat/SE_bhat
print("tstat of intercept = %0.2f and tstat of slope = %0.2f" % (tstat0, tstat1))

# critical value for t statistics
t_critical = stats.t.ppf(1-0.025, n-2)
print("critical value for two-tail t test = %0.2f" % (t_critical))

# two-sided pvalue 
pvalue0 = stats.t.sf(np.abs(tstat0), n-2)*2
pvalue1 = stats.t.sf(np.abs(tstat1), n-2)*2
print("p values = %0.2f percent and %0.2f precent" % (pvalue0*100, pvalue1*100))

# F statistic
msr = sum((yf-ya)**2)
F = msr/sigma_e2
print("F statistic = %0.2f" % (F))

# critical value for F statistic
F_critical = stats.f.ppf(1-0.025, 1, n-2)
print("critical value for two-tail F test = %0.2f" % (F_critical))
pvalue = stats.f.sf(F, 1, n-2)
print("p value of Fstat = %0.2f%c" % (pvalue*100, 37))

# plot the fitted line and data
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

xl = np.linspace(0, 300, 200)
yl = ahat + bhat*xl
plt.plot(xl, yl)
plt.plot(x, y, 'o')
plt.text(21, 2.95, "Gold",    fontdict=font)
plt.text(130, 2.25, "Silver",    fontdict=font)
plt.text(240, 2.1, "Iron",    fontdict=font)
plt.xlabel('Word Number', fontdict=font)
plt.ylabel("Log Density in g/cm$^3$", fontdict=font)
plt.grid()
plt.show()

