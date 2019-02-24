# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:17:38 2018

@author: Harish Reddy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm
from math import log, sqrt, exp
from scipy.optimize import least_squares

google_call = pd.read_csv('goog_call.csv',parse_dates=['date','expiry'])
google_put = pd.read_csv('goog_put.csv',parse_dates=['date','expiry'])
google_call['call']=(google_call.loc[:,'best_bid']+google_call.loc[:,'best_offer'])/2
google_put['put']=(google_put.loc[:,'best_bid']+google_put.loc[:,'best_offer'])/2

def Black76LognormalCall(S, K, r, sigma,T):
    d1 = (log(S/K)+(r+sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def impliedCallVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -
                        Black76LognormalCall(S, K, r, x, T),
                        1e-6, 1)
    return impliedVol

def Black76LognormalPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)


def impliedPutVolatility(S, K, r, price, T):
    impliedVol = brentq(lambda x: price -
                        Black76LognormalPut(S, K, r, x, T),
                        1e-6, 1)

    return impliedVol

def BachelierCall(S, K, r, sigma, T):
    d = (S-K)/(S*sigma*np.sqrt(T))
    return (S-K)*norm.cdf(d) + (S*sigma*np.sqrt(T)*norm.pdf(d))

def BachelierPut(S, K, r, sigma, T):
    d = (K-S)/(S*sigma*np.sqrt(T))
    return (S-K)*norm.cdf(d) - (S*sigma*np.sqrt(T)*norm.pdf(d))

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], 0.8, x[1], x[2]))**2

    return err

S = 846.9
K=850
r=0
Total_days = (google_call.iloc[0,1]-google_call.iloc[0,0]).days
T=Total_days/365
   

p= google_call.loc[google_call['strike'] == K]['call'].tolist()[0]
sigma = impliedCallVolatility(S, K, r, p, T)
F = S*np.exp(r*T)

strikes=google_call['strike'].tolist()

summary = []
for K in strikes:
    if K >= S :
        price = BachelierCall(S, K, r, sigma, T)
        impliedvol_N = impliedCallVolatility(S, K, r, price, T)
        price = Black76LognormalCall(S, K, r, sigma, T)
        impliedvol_LN = impliedCallVolatility(S, K, r, price, T)
    else:
        price = -BachelierPut(S, K, r, sigma, T)
        impliedvol_N = impliedPutVolatility(S, K, r, price, T)
        price = Black76LognormalPut(S, K, r, sigma, T)
        impliedvol_LN = impliedPutVolatility(S, K, r, price, T)

    summary.append([K, impliedvol_N, impliedvol_LN])

df = pd.DataFrame(summary, columns=['strike', 'Normal IV', 'Lognormal IV'])

for i in range(len(df)):
    K = df.iloc[i,0]
    if K < S:
        df.loc[i,'market'] = impliedPutVolatility(S, K, r, google_put.loc[i,'put'], T)
    else:
        df.loc[i,'market'] = impliedCallVolatility(S, K, r, google_call.loc[i,'call'], T)
        
betas=[0.2,0.4,0.6,0.8]        
for i in betas:
    beta = i
    c='beta=' + str(np.round(beta,1))
    for i in range(len(df)):
        K = df.iloc[i,0]
        if K < S:
            price=Black76LognormalPut(S/beta, (K+(1-beta)*S/beta), r, sigma*beta, T)
            df.loc[i,c] = impliedPutVolatility(S, K, r, price, T)
        else:
            price=Black76LognormalCall(S/beta, (K+(1-beta)*S/beta), r, sigma*beta, T)
            df.loc[i,c] = impliedCallVolatility(S, K, r, price, T)
            
plt.figure(figsize=[8,6])           
plt.plot(df['strike'],df['Normal IV'],linewidth=3.0)
plt.plot(df['strike'],df['Lognormal IV'],linewidth=3.0)
plt.plot(df['strike'],df['market'],'bo',linewidth=3.0)
i=4 #4 is the column number of beta=0.2
plt.plot(df['strike'],df[df.columns[i]],linewidth=3.0)
i=i+1
plt.plot(df['strike'],df[df.columns[i]],linewidth=3.0)
i=i+1
plt.plot(df['strike'],df[df.columns[i]],linewidth=3.0)
i=i+1
plt.plot(df['strike'],df[df.columns[i]],linewidth=3.0)
plt.axis([google_call.loc[0,'strike'],google_call.loc[len(google_call.loc[:,'strike'].tolist())-1,'strike'], 0.1875, 0.425])
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.legend()
plt.show()

initialGuess = [0.02, 0.2, 0.1]
res = least_squares(lambda x: sabrcalibration(x,
                                              df['strike'].values,
                                              df['market'].values,
                                              F,
                                              T),
                    initialGuess)
alpha = res.x[0]
beta = 0.8
rho = res.x[1]
nu = res.x[2]

for i in range(len(df)):
    df.loc[i, 'SABR IV'] = SABR(S, df.iloc[i, 0], T, alpha, beta, rho, nu)

plt.figure(figsize=[8,6])     
plt.plot(df['strike'],df['Normal IV'],linewidth=3.0)
plt.plot(df['strike'],df['Lognormal IV'],linewidth=3.0)
plt.plot(df['strike'],df['market'],'bo',linewidth=3.0)
plt.plot(df['strike'],df['beta=0.4'],linewidth=3.0)
plt.plot(df['strike'],df['SABR IV'],linewidth=3.0)
plt.axis([google_call.loc[0,'strike'],google_call.loc[len(google_call.loc[:,'strike'].tolist())-1,'strike'], 0.1875, 0.425])
plt.xlabel('Strikes')
plt.ylabel('Implied volatility')
plt.legend()
plt.show()

print('alpha:',alpha)
print('rho:',rho)
print('nu:',nu)
      
