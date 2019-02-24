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
import statsmodels.api as sm
from scipy.integrate import quad

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








print('--------------------------------------------------')
print('Part 3: Static Replication')

S=846.9
T=505
F=S*np.exp(r*T)

discount=pd.read_csv('discount.csv')
model1=sm.OLS(discount.iloc[10:15,1],discount.iloc[10:15,0]).fit()
slope=model1.params.values

r=(discount.iloc[11,1]+slope*(T-discount.iloc[11,0]))/100
T=505/365
sigma_p3=SABR(S, S, T, alpha, beta, rho, nu)
sigma_p3=0.26

print(r,T,sigma_p3,S)

def V_ST3Black(r,T,sigma_p3,S):
    num1=np.exp(-r*T)
    num2=S**3*np.exp(3*T*(r+sigma_p3**2))
    num3=(r-(sigma_p3**2)/2)*T+np.log(S)
    return num1*(num2+2.5*num3+10)

print('Value of derivative under Black Scholes is:',V_ST3Black(r,T,sigma_p3,S)[0])

print('--------------------------------------------------')

def V_ST3Bachelier(S,T,r,sigma):
    num1=S**3+3*sigma**2*S**3*T
    func=lambda x:log(1+sigma*np.sqrt(T)*x)*exp(-0.5*(x**2))
    bound=-1/(sigma*np.sqrt(T))
    InteBache=quad(func,bound,np.inf)[0]
    num2=np.log(S)+InteBache
    return np.exp(-r*T)*(num1+2.5*num2+10)

print('Value of derivative under Bachelier is:',V_ST3Bachelier(S,T,r,sigma_p3)[0])


print('--------------------------------------------------')
print('Static Replication of ST3 Payoff')

F=S*np.exp(r*T)
print('F:',F,'T:', T, 'alpha:',alpha, 'beta:',beta, 'rho:',rho, 'nu:',nu)
def h(x):
    return x**3+2.5*np.log(x)+10

def h1prime(x):
    return 3*x**2+2.5/S

def h2prime(x):
    return 6*x-2.5/(x**2)

def callintegrandLog(K, S, r, T, sigma):
    price = Black76LognormalCall(S, K, r,sigma,T)*h2prime(K)
    return price

def putintegrandLog(K, S, r, T, sigma):
    price = Black76LognormalPut(S, K, r, sigma, T)*h2prime(K)
    return price

I_put1 = quad(lambda x: putintegrandLog(x, S, r, T, SABR(F, x, T, alpha, beta, rho, nu
                                                         )), 0.0, F)
I_call1 = quad(lambda x: callintegrandLog(x, S, r, T, SABR(F, x, T, alpha, beta, rho, nu
                                                           )), F, 5000)
V_repli=np.exp(-r*T)*h(F)+I_put1[0]+I_call1[0]

print('Value of derivative under Static replication is:',V_repli[0])

print('--------------------------------------------------')
print('Model free integrated variance')

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
    return BlackScholesCall(S, K, r, sigma, T) - S + K*np.exp(-r*T)

def callintegrand(K, S, r, T, sigma):
    price = BlackScholesCall(S, K, r, sigma, T) / K**2
    return price


def putintegrand(K, S, r, T, sigma):
    price = BlackScholesPut(S, K, r, sigma, T) / K**2
    return price

F = S * np.exp(r*T)
I_put2 = quad(lambda x: putintegrand(x, S, r, T, sigma), 0.0, F)
I_call2 = quad(lambda x: callintegrand(x, S, r, T, sigma), F, np.inf    )

E_var_Black = 2*np.exp(r*T)*(I_put2[0] + I_call2[0])
print('The expected integrated variance under Black Scholes monte carlo is: %.9f' % E_var_Black)

E_var_Black_Manual=sigma_p3**2*T
print('The expected integrated variance under Black Scholes (Manual) is: %.9f' % E_var_Black_Manual)


print('--------------------------------------------------')
print('Bachelier expected integrated variance')

def BachelierCall(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return (S-K)*norm.cdf(d1) + sigma*S*np.sqrt(T)*norm.pdf(d1)

def BachelierPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return (K-S)*norm.cdf(-d1) - S*sigma*np.sqrt(T)*norm.pdf(-d1)


def callintegrand1(K, S, T, sigma):
    price = BachelierCall(S, K, sigma, T) / K**2
    return price


def putintegrand1(K, S, T, sigma):
    BachelierPut(S, K, sigma, T) / K**2
    return price

I_put3 = quad(lambda x: putintegrand1(x, S, T, sigma), 0.0, F)
I_call3 = quad(lambda x: callintegrand1(x, S, T, sigma), F, np.inf)


E_var_Bachelier = 2*np.exp(r*T)*(I_put3[0] + I_call3[0])
print('The expected integrated variance under Bachelier is: %.9f' % E_var_Bachelier)

print('--------------------------------------------------')
print('SABR expected integrated variance')

def callintegrandswap(K, S, r, T, sigma):
    price = BlackScholesCall(S, K, r, sigma, T) / K**2
    return price


def putintegrandswap(K, S, r, T, sigma):
    price = BlackScholesPut(S, K, r, sigma, T) / K**2
    return price

I_put4 = quad(lambda x: putintegrandswap(x, S, r, T, SABR(F, x, T, alpha, beta, rho, nu
                                                         )), 0.0, F)
I_call4 = quad(lambda x: callintegrandswap(x, S, r, T, SABR(F, x, T, alpha, beta, rho, nu
                                                           )), F, 5000)

E_var_Static = 2*np.exp(r*T)*(I_put4[0] + I_call4[0])
print('The expected integrated variance under Static Replication is: %.9f' % E_var_Static)

