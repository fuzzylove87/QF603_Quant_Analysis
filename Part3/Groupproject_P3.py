"""
Date: 2018-07-13, 2018-07-14, 2018-11-09
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import stats

def get_betas(x,y):
    X = np.asmatrix(x)
    X = np.c_[np.ones(len(y)), x]
    Xt = X.transpose()
    Z = Xt.dot(X)
    Zi = inv(Z)
    return (Zi.dot(Xt.dot(y)),Zi,X)

def get_RSS(beta_hat,x,y):
    epsilon = y - x.dot(beta_hat)
    RSS = epsilon.transpose().dot(epsilon)
    return RSS

def get_TSS(y):
    my = np.mean(y)
    disp_y = y - my
    disp_yt = disp_y.transpose()
    TSS = float(disp_yt.dot(disp_y))
    return TSS

def get_R2_AdjustedR2 (RSS,TSS,n,K):
    R2 = float(1 - RSS/TSS)
    R_bar2 = float(1-((RSS/(n-K))/(TSS/(n-1))))
    return R2,R_bar2

def get_t_stats(RSS,n,k,beta_hat,Zi):
    vr = RSS/(n-k)
    V = Zi*float(vr)
    se = np.sqrt(V.diagonal())
    t = np.zeros(k, dtype=float)
    for i in range(k):
        t[i] = float(beta_hat[i])/se[i]
        
    return t

def get_AIC(RSS,n,k):
    vr = RSS/(n-k)
    vr = float(vr)
    return  ((n*np.log(RSS/n)+ 2*k),np.log(vr)+ 2*k/n)

#F-test stats
def get_F_stats(RSS,URSS,T,K,m):
    return ((RSS-URSS)/URSS) * (((T)-K)/m)


# Reading FF 5 factors
fn = 'F-F_Research_Data_5_Factors_2x3.CSV'
FF = pd.read_csv(fn, skiprows=3, nrows = 663)
FF = FF.rename(columns={"Unnamed: 0": "Date"})
FF.index = pd.to_datetime(FF.Date, format = '%Y%m').dt.to_period('m')
FF = FF.drop(['Date'], axis=1)
FF = FF*1e-2

# Reading S&P500 CAPM
fn = '^GSPC.csv'
dji = pd.read_csv(fn)
dji = dji.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
dji.index = pd.to_datetime(dji.Date)
dji = dji.resample('M').last()
dji = dji.drop(['Date'], axis=1)
ri = dji.pct_change()[1:]
ri.index = ri.index.to_period('m')
ri = ri.rename(columns={'Close': 'Return'})

# Align monthly data/ Data cleansing
df = pd.concat([ri, FF], axis=1)
df = df.dropna()

#Assigning CAPM and FF returns to variables 
y = df.Return - df.RF
X = df.iloc[:, 1:2]
x_FF =  df.iloc[:, 1:6]

sample_size = df.shape[0]

y = np.asmatrix(y)
y = y.transpose()

#betas
CAPM_beta,Zi_CAPM,x_CAPM_One = get_betas(X,y)
FF_beta,Zi_FF,x_FF_One = get_betas(x_FF,y)
#RSS
RSS_FF = get_RSS(FF_beta,x_FF_One,y)
RSS_CAPM = get_RSS(CAPM_beta,x_CAPM_One,y)

#t-stats
t_FF = get_t_stats(RSS_FF,sample_size,FF_beta.shape[0],FF_beta,Zi_FF)

#t-critcal values
t_critical_value = stats.t.ppf(1-0.025, sample_size-len(FF_beta))

#R square and adjusted R square
TSS = get_TSS(y)
R2,R_bar2 = get_R2_AdjustedR2 (RSS_FF,TSS,sample_size,FF_beta.shape[0])

#AIC FF
AIC_FF_notes,AIC_FF_mlr = get_AIC(RSS_FF,sample_size,FF_beta.shape[0])

f_test = get_F_stats(RSS_CAPM,RSS_FF,sample_size,FF_beta.shape[0],4)

#Adding graphs


print("Task 5\n")

print("1.FF 5 factors: a_hat: %0.5f , b_hat: %0.5f, c_hat: %0.5f, d_hat: %0.5f, e_hat: %0.5f, f_hat: %0.5f" % (FF_beta[0],FF_beta[1],FF_beta[2],FF_beta[3],FF_beta[4],FF_beta[5]))
print("1.CAPM factors: a_hat: %0.5f , b_hat: %0.5f" % (CAPM_beta[0],CAPM_beta[1]))
print("2.t_stats: a_hat:%0.5f , b_hat: %0.5f, c_hat: %0.5f, d_hat: %0.5f, e_hat: %0.5f, f_hat: %0.5f" % (t_FF[0],t_FF[1],t_FF[2],t_FF[3],t_FF[4],t_FF[5]))
print("3.Critical values for 5 percent significance level: +%0.5f , -%0.5f" % (t_critical_value,t_critical_value))
print("4.R_bar and Adjusted R_bar: %0.5f , %0.5f" % (R2,R_bar2))
print("5.AIC from notes : %0.5f; AIC from mlr0.py: %0.5f " % (AIC_FF_notes,AIC_FF_mlr))

print("\nTask 6\n")
print("1.F-test statistics: %0.5f " % f_test)

















