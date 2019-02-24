# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:30:56 2018

@author: ykfri
"""
import scipy.optimize as solver
'''
Num=100000
sds=[]
rtn=[]
mean=IP.mean()
cov=IP.cov()
for i in range(Num):
    w=1/np.random.random(10)
    w /= sum(w)
    rtn.append(sum(mean * w))
    sds.append(np.sqrt(np.dot(np.dot(w,cov),w.T)))  
plt.plot(sds,rtn,'ko')

x0=np.array([1.0/10 for x in range(10)])
bounds=tuple((0,1) for x in range(10))
'''

def sd(w):
    return np.sqrt(np.dot(w, np.dot(cov, w.T)))

given_r=np.arange(0.50,1.20,0.001)
risk=[]
for i in given_r:
    constraints=[{'type':'eq','fun':lambda x: sum(x)-1},
                 {'type':'eq','fun':lambda x: sum(x*mean)-i}]
    outcome=solver.minimize(sd,x0=x0,constraints=constraints,bounds=bounds)
    risk.append(outcome.fun)
    
plt.plot(risk,given_r,'x')

