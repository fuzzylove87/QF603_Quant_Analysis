# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:59:13 2018

@author: ChanJung Kim
"""

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

MP = pd.read_excel('Market_Portfolio.xlsx', 'Sheet1') # Import market portfolio
IP = pd.read_excel('Industry_Portfolios.xlsx', 'Sheet1') # Import industry portfolio

IP = IP.iloc[:, 1:11]
RM = pd.DataFrame(MP['Market'])
RF = 0.13

X = RM - RF
Y = IP - RF

# Q1
regr1 = LinearRegression()
regr1.fit(X, Y)

A = pd.DataFrame(regr1.intercept_, columns = ['Alpha'])
B = pd.DataFrame(regr1.coef_, columns = ['Beta'])
AB = pd.concat((A, B), axis=1)
AB.index = IP.columns

# Q2
R_SML = pd.concat((IP.mean(), RM.mean()))
B_SML = B.iloc[:, :]
B_SML.loc[10]=["1"]

regr2 = LinearRegression()
regr2.fit(B_SML, R_SML)
Intercept_SML = float(regr2.intercept_)
Slope_SML = float(regr2.coef_)

# Q3
Beta_Random = [i/100 for i in range(0, 201)]
Est_Return = [i * Slope_SML + Intercept_SML for i in Beta_Random]

# Result print
print('Alpha and Beta for each industry is as below')
print(AB)
print('Mean return of industry and market portfolio is as below')
print(R_SML)
print('-----------------------------------------------------------')
print('Y-intercept and slope for SML is as below')
print('Y-intercept : ' + str(Intercept_SML))
print('Slope : ' + str(Slope_SML))

plt.plot(Beta_Random, Est_Return, label='Security Market Line',color='b')
plt.plot(B_SML, R_SML, 'ko', label='Industry and Market Portfolio')
plt.text(0.01,1.03,'Rf',fontsize=12)
plt.xlabel('Beta')
plt.ylabel('Expected Return (%)')
plt.grid(True)
plt.legend(loc='upper left')
plt.xlim(0, 2)
plt.savefig('Security_Market_Line.png', dpi =800)
plt.show()
