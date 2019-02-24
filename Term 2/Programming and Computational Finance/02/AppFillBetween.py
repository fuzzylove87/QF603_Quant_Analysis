# fill_between
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=[12,8] # (optional)
data=pd.read_csv('CC3.SI.csv',index_col=0,parse_dates=True)
data.drop(data.index[data['Volume']==0],inplace=True)
ma=data['Close'].rolling(15).mean()
mstd=data['Close'].rolling(15).std()
plt.plot(data.index, data['Close'], 'k')
plt.plot(ma.index, ma, 'b')
plt.fill_between(mstd.index, ma-2*mstd, ma+2*mstd,
                 color='b', alpha=0.2)
plt.show()
