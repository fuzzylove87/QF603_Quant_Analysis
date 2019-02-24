# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:02:13 2018

@author: ykfri
"""

from math import log, sqrt, exp
from scipy import stats

class call_option(object):
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = float(S0)
        self.K= K
        self.T= T
        self.r= r
        self.sigma= sigma

    def value(self):
        d1 = ((log(self.S0 / self.K) + (self.r+ 0.5 * self.sigma** 2) * self.T)
                / (self.sigma* sqrt(self.T)))
        d2 = ((log(self.S0 / self.K) + (self.r-0.5 * self.sigma** 2) * self.T)
                / (self.sigma* sqrt(self.T)))
        value = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0)
            -self.K* exp(-self.r* self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return value

    def vega(self):
        d1 = ((log(self.S0 / self.K) + (self.r+ 0.5 * self.sigma** 2) * self.T)
            / (self.sigma* sqrt(self.T)))
        vega= self.S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(self.T)
        return vega

    def imp_vol(self, C0, sigma_est=0.2, it=100):
        option = call_option(self.S0, self.K, self.T, self.r, sigma_est)
        for i in range(it):
            option.sigma = (option.value() -C0) / option.vega()
            return option.sigma

o=call_option(100., 105., 1.0, 0.05, 0.2)
print(o.value())
print(o.vega())
print(o.imp_vol(C0=o.value()))