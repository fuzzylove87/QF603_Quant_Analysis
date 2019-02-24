# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:47:21 2018

@author: ykfri
https://thequantmba.wordpress.com/2016/04/16/modern-portfolio-theory-python/
"""
# import module
from __future__ import division
from matplotlib import pyplot as plt
from numpy.linalg import inv,pinv
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# import data
df = pd.read_excel('Industry_Portfolios.xlsx', 'Sheet1')
df1 = df.iloc[:, 1:11]

# mean and variance
mean_table = DataFrame(df1.mean(), columns = ['Expected Return'])
var_table = DataFrame(df1.std(), columns = ['Standard Deviation'])
V = np.matrix(df1.cov())
R = np.matrix(df1.mean())
R = R.reshape(R.size, 1)

rf = 0.13
I = np.ones((len(R),1)) #np.ones is array
VAR = np.diag(V)
SD = np.sqrt(np.diag(V))

# Alpha, Zeta, Delta, and Denominator for Lagrange Multipliers
Delta = I.T*pinv(V)*I # in Matrix format, the multiplication is applied on Matrix-wise
Alpha = R.T*pinv(V)*I
Zeta = R.T*pinv(V)*R
D = Zeta*Delta-Alpha**2 #Denominator of Lagrange multipliers(lambda, gamma)

# Set the range of the TARGET RETURN
mu = np.arange(0,2,2/10000)

# Print mean, variance, and covariance
meanvar_table = pd.concat([mean_table, var_table], axis=1)
print(meanvar_table)
print(V)

minvar = (Zeta-2*Alpha*mu+(Delta*mu**2))/D;
minstd = np.sqrt(minvar)[0]
minvar = np.squeeze(np.asarray(minvar))
minstd = np.squeeze(np.asarray(minstd))  
#squeeze removes one-dimensional entry from the shape of the given array. 
# asarray copy the given value when it's not a ndarray. Here, minstd is matrix, and minstd is copiedf

plt.plot(minvar,mu)
plt.title('Efficient frontier in Mean-Var space',fontsize=20)
plt.ylabel('Expected Return (%)', fontsize=14)
plt.xlabel('Variance (%)', fontsize=14)
plt.show()


mu_g = Alpha/Delta #the Value of Rp in Rmv
var_g = 1/Delta #the value of Sigma^2 in Rmv
std_g = np.sqrt(var_g)

# Minimum Variance Portfolio Weights
w_g = (pinv(V)*I)/Delta # infer from mean return 

# Plot Efficient Frontier
plt.plot(minstd,mu)
plt.plot(std_g, mu_g, '*')
plt.text(0.1+std_g,mu_g,'Rmv',fontsize=12)
plt.title('Efficient frontier in Mean-SD space',fontsize=20)
plt.ylabel('Expected Return (%)', fontsize=14)
plt.xlabel('Standard Deviation (%)', fontsize=14)
plt.show()

#TANGENT PORTFOLIO
# Expected Return of Tangency Portfolio
mu_tan = (Alpha*rf-Zeta)/(Delta*rf-Alpha);

# Variance and Standard Deviation of Tangency Portfolio
vartan = (Zeta-2*rf*Alpha + (rf**2*Delta))/((Alpha-Delta*rf)**2);
stdtan = np.sqrt(vartan);

 # Weights for Tangency Portfolio
w_tan = (pinv(V)*(R - rf*I))/(Alpha-Delta*rf)

# Tangency Line
m_tan = mu[mu >= rf];
minvar_rf = (m_tan-rf)**2/(Zeta-2*rf*Alpha+Delta*rf**2);
minstd_rf = np.sqrt(minvar_rf);
minstd_rf = np.squeeze(np.asarray(minstd_rf))

# Plot with tangency portfolio
plt.plot(minstd,mu,minstd_rf,m_tan, 'r', std_g,mu_g,'bo', stdtan,mu_tan, 'bo', rf, 'bo')
plt.title('Efficient frontier with Riskless asset',fontsize=18)
plt.ylabel('Expected Return (%)', fontsize=12)
plt.xlabel('Standard Deviation (%)', fontsize=12)
plt.text(0.5,rf,'rf',fontsize=12)
plt.text(0.5+std_g,mu_g,'Globalimum Variance Portfolio',fontsize=12);
plt.text(0.5+stdtan,mu_tan,'Tangency portfolio',fontsize=12);
plt.xlim(0,10)
plt.ylim(0, 2)
plt.show()

print(w_tan)