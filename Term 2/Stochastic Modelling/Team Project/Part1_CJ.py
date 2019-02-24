# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:13:58 2018

@author: ykfri
"""

import numpy as np
from scipy.stats import norm

# 1. Black-Scholes Model
# Vanilla call/put 
def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Digital cash-or-nothing call/put
def BlackScholesCashCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesCashPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2)

# Digital asset-or-nothing call/put
def BlackScholesAssetCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1)

def BlackScholesAssetPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(-d1)

# Bachelier Model
# Vanila call/put
def BachelierCall(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return S*(1+sigma*np.sqrt(T))*norm.cdf(d1) - K*norm.pdf(d1)

def BachelierPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return K*norm.pdf(-d1) + S*(sigma*np.sqrt(T)-1)*norm.cdf(-d1)

# Digital cash-or-nothing call/put
def BachelierCashCall(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return K*norm.pdf(d1)

def BachelierCashPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return K*norm.pdf(-d1)

# Digital asset-or-nothing call/put
def BachelierAssetCall(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return S*(1+ sigma*np.sqrt(T))*norm.cdf(d1)

def BachelierAssetPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return S*(sigma*np.sqrt(T)-1)*norm.cdf(-d1)

# 3. Black76 Model
#Vanila call/put
def Black76Call(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))

def Black76Put(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

# Digital cash-or-nothing call/put
def Black76CashCall(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*K*norm.cdf(d2)

def Black76CashPut(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*K*norm.cdf(-d2)

# Digital asset-or-nothing call/put
def Black76AssetCall(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*F*norm.cdf(d1)

def Black76AssetPut(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*F*norm.cdf(-d1)

# 4. Displaced-diffusion Model
#Vanila call/put
def DiffusionCall(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))   
    
def DiffusionPut(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))

# Digital cash-or-nothing call/put
def DiffusionCashCall(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*K*norm.cdf(d2)
    
def DiffusionCashPut(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*K*norm.cdf(-d2)

# Digital asset-or-nothing call/put
def DiffusionAssetCall(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*F*norm.cdf(d1)
    
def DiffusionAssetPut(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*F*norm.cdf(-d1)

S = 1403
K = 1350
r = 0.0534
sigma = 0.26
T = 0.102777778
print(BlackScholesCall(S, K, r, sigma, T))
print(BachelierCall(S, K, sigma, T))
print(BachelierCashCall(S, K, sigma, T))