# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:33:38 2018

@author: ykfri
"""

# Moving Average Crossover 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["figure.figsize"]=[12,8] # (optional)
data=pd.read_csv('CC3.SI.csv',index_col=0,parse_dates=True)
data.drop(data.index[data['Volume']==0],inplace=True)
data['15d']= np.round(data['Adj Close'].rolling(window=15).mean(),3)
data['50d']= np.round(data['Adj Close'].rolling(window=50).mean(),3)
x=data['15d']-data['50d']
x[x>0]=1
x[x<=0]=0
y=x.diff()
idxSell=y.index[y<0]
idxBuy=y.index[y>0]
data['crossSell']=np.nan
data.loc[idxSell,'crossSell']=data.loc[idxSell,'Adj Close']
data['crossBuy']=np.nan
data.loc[idxBuy,'crossBuy']=data.loc[idxBuy,'Adj Close']
fig, ax = plt.subplots()
data[['Adj Close', '15d', '50d','crossSell','crossBuy']].plot(
        ax=ax,
        style=['k-','b-','c-','ro','yo'],
        linewidth=1)
plt.autoscale(enable=True, axis='x', tight=True)
plt.autoscale(enable=True, axis='y', tight=True)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.show()