# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 14:00:06 2018

@author: ykfri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

data = pd.read_csv('dataset01.csv', header=0)
print(data)
print(len(data))

def option_BS(data):
    S,K=data['S'],data['K']
    r,q=data['r'],data['q']
    sigma=data['sigma']
    T=data['T']
    d1=(np.log(S/K)+(r-q+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*ss.norm.cdf(
            d1) - K*np.exp(-r*T)*ss.norm.cdf(d2)

data['BS'] = data.apply(option_BS,axis=1)

print(data)
