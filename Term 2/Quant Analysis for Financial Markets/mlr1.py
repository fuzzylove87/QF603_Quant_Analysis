"""
Date: 2018-07-14
https://stackoverflow.com/questions/23420454/newey-west-standard-errors-for-ols-in-python
"""

from __future__ import division, print_function

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.DataFrame({'y':[1,3,5,7,4,5,6,4,7,8,9],
                   'x':[3,5,6,2,4,6,7,8,7,8,9]})

reg = smf.ols('y ~ 1 + x',data=df).fit()
print(reg.summary())

results = reg.get_robustcov_results(cov_type='HAC',maxlags=1)
print (results.summary())




