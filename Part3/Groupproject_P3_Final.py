"""
Date: 2018-07-13, 2018-07-14, 2018-11-09
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import stats
from matplotlib import pyplot as plt

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
    return  (n*np.log(RSS/n)+ 2*k)

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
t_CAPM = get_t_stats(RSS_CAPM,sample_size,CAPM_beta.shape[0],CAPM_beta,Zi_CAPM)

#t-critcal values
t_critical_value = stats.t.ppf(1-0.025, sample_size-len(FF_beta))

#R square and adjusted R square
TSS = get_TSS(y)
R2,R_bar2 = get_R2_AdjustedR2 (RSS_FF,TSS,sample_size,FF_beta.shape[0])
R2_CAPM,R_bar2_CAPM = get_R2_AdjustedR2 (RSS_CAPM,TSS,sample_size,CAPM_beta.shape[0])

#AIC FF
AIC_FF = get_AIC(RSS_FF,sample_size,FF_beta.shape[0])
AIC_CAPM = get_AIC(RSS_CAPM,sample_size,CAPM_beta.shape[0])

f_test = get_F_stats(RSS_CAPM,RSS_FF,sample_size,FF_beta.shape[0],4)

#Adding graphs
FF_Y_hat = np.array(x_FF_One.dot(FF_beta))
FF_Y_hat_chart = list(FF_Y_hat.reshape(1, -1))
X_chart = np.array(X).reshape(-1, 1)
y = np.array(y)
y_chart = list(y)

Pseudo_X = np.arange(-0.25, 0.15, 0.005)
CAPM_Y = float(CAPM_beta[0]) + Pseudo_X*float(CAPM_beta[1])

FF_slope = sum(FF_Y_hat*y)/sum(FF_Y_hat*FF_Y_hat)
FF_Y = Pseudo_X*float(FF_slope) + float(FF_beta[0])

plt.scatter(X_chart, y_chart, color='k', s=4, label = 'Scatter plot of two variables')
plt.plot(Pseudo_X, CAPM_Y, label='Regression Line')
plt.title('CAPM',fontsize=15)
plt.ylabel('Expected Return (%)', fontsize=12)
plt.xlabel('Mkt-RF (%)', fontsize=12)
plt.legend()
#plt.savefig('CAPM_regression.jpeg', dpi=800,quality=100)
plt.show()

plt.scatter(FF_Y_hat_chart, y_chart, color='r', s=4, label = 'Scatter plot of two variables')
plt.plot(Pseudo_X, FF_Y, label = 'Regression Line')
plt.title('Fama French Five-Factor Model',fontsize=15)
plt.ylabel('Expected Return (%)', fontsize=12)
plt.xlabel('Predicted Dependent Variable (%)', fontsize=12)
plt.legend()
#plt.savefig('FF_regression.jpeg', dpi=800,quality=100)
plt.show()

print("Task 5\n")

print("1.FF 5 factors: a_hat: %0.5f , b_hat: %0.5f, c_hat: %0.5f, d_hat: %0.5f, e_hat: %0.5f, f_hat: %0.5f" % (FF_beta[0],FF_beta[1],FF_beta[2],FF_beta[3],FF_beta[4],FF_beta[5]))
print("1.CAPM factors: a_hat: %0.5f , b_hat: %0.5f" % (CAPM_beta[0],CAPM_beta[1]))
print("2.t_stats: a_hat:%0.5f , b_hat: %0.5f, c_hat: %0.5f, d_hat: %0.5f, e_hat: %0.5f, f_hat: %0.5f" % (t_FF[0],t_FF[1],t_FF[2],t_FF[3],t_FF[4],t_FF[5]))
print("2.t_stats: a_hat:%0.5f , b_hat: %0.5f" % (t_CAPM[0],t_CAPM[1]))
print("3.Critical values for 5 percent significance level: +%0.5f , -%0.5f" % (t_critical_value,t_critical_value))
print("4.R_bar and Adjusted R_bar for FF: %0.5f , %0.5f" % (R2,R_bar2))
print("4.R_bar and Adjusted R_bar for CAPM: %0.5f , %0.5f" % (R2_CAPM,R_bar2_CAPM))
print("5.AIC for FF: %0.5f" % AIC_FF)
print("5.AIC from CAPM: %0.5f" % (AIC_CAPM))

print("\nTask 6\n")
print("1.F-test statistics: %0.5f " % f_test)













