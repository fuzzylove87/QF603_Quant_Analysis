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
    return np.exp(-r*T)*norm.cdf(d2)

def BlackScholesCashPut(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(-d2)

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
    return (S-K)*norm.cdf(d1) + sigma*S*np.sqrt(T)*norm.pdf(d1)

def BachelierPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return (K-S)*norm.cdf(-d1) - S*sigma*np.sqrt(T)*norm.pdf(-d1)

# Digital cash-or-nothing call/put
def BachelierCashCall(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return norm.cdf(d1)

def BachelierCashPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return norm.cdf(-d1)

# Digital asset-or-nothing call/put
def BachelierAssetCall(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return S*norm.cdf(d1) + sigma*S*np.sqrt(T)*norm.pdf(d1)

def BachelierAssetPut(S, K, sigma, T):
    d1 = (S-K)/(S*sigma*np.sqrt(T))
    return S*norm.cdf(-d1) + sigma*S*np.sqrt(T)*norm.pdf(-d1)

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
    return np.exp(-r*T)*norm.cdf(d2)

def Black76CashPut(S, K, r, sigma, T):
    F = S*np.exp(r*T)
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(-d2)

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
    return np.exp(-r*T)*norm.cdf(d2)
    
def DiffusionCashPut(S, K, r, sigma, T, beta):
    F = S*np.exp(r*T)
    F = F/beta
    K = K + (1-beta)*F
    sigma = sigma*beta
    d1 = (np.log(F/K)+(sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return np.exp(-r*T)*norm.cdf(-d2)

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
r = 0.0534
sigma = 0.26
T = 0.102777778
beta = 1
K = 1350

print('1. Black-Scholes ------------------------------------------------')
print(BlackScholesCall(S, K, r, sigma, T))
print(BlackScholesPut(S, K, r, sigma, T))
print(BlackScholesCashCall(S, K, r, sigma, T))
print(BlackScholesCashPut(S, K, r, sigma, T))
print(BlackScholesAssetCall(S, K, r, sigma, T))
print(BlackScholesAssetPut(S, K, r, sigma, T))

print('2. Bachelier------------------------------------------------')
print(BachelierCall(S, K, sigma, T))
print(BachelierPut(S, K, sigma, T))
print(BachelierCashCall(S, K, sigma, T))
print(BachelierCashPut(S, K, sigma, T))
print(BachelierAssetCall(S, K, sigma, T))
print(BachelierAssetPut(S, K, sigma, T))

print('3. Black76------------------------------------------------')
print(Black76Call(S, K, r, sigma, T))
print(Black76Put(S, K, r, sigma, T))
print(Black76CashCall(S, K, r, sigma, T))
print(Black76CashPut(S, K, r, sigma, T))
print(Black76AssetCall(S, K, r, sigma, T))
print(Black76AssetPut(S, K, r, sigma, T))

print('4. Diffusion------------------------------------------------')
print(DiffusionCall(S, K, r, sigma, T, beta))
print(DiffusionPut(S, K, r, sigma, T, beta))
print(DiffusionCashCall(S, K, r, sigma, T, beta))
print(DiffusionCashPut(S, K, r, sigma, T, beta))
print(DiffusionAssetCall(S, K, r, sigma, T, beta))
print(DiffusionAssetPut(S, K, r, sigma, T, beta))