# candlestick_ohlc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import date2num
plt.rcParams["figure.figsize"]=[12,8] # (optional)
data=pd.read_csv('CC3.SI.csv',index_col=0,parse_dates=True)
data.drop(data.index[data['Volume']==0],inplace=True)
r=data.iloc[:15, :]
fig, ax = plt.subplots()
d=date2num(r.index.date)
candlestick_ohlc(ax, zip(d, r.Open, r.High, r.Low, r.Close), 
                 width=0.5, colorup='g', colordown='r', alpha=1)
plt.setp(ax.get_xticklabels(), rotation=30)
ax.xaxis_date()
plt.show()
