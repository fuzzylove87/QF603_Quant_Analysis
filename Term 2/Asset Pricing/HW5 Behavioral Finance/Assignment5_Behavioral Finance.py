import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# empty lists to store epsilon nu and consumption growth of consumption growth
epsilon = [] #Standard normal random variable
growth= [] #Consumption Growth


#****************************************************************************************************
#Parameter Choices for the Investor's Utility Function

delta=0.99 #Subjective Discount Factor
gamma=1 #gamma>0 is coefficient of relative risk aversion for consumption shocks
lamb=2 # Loss Aversion makes investor more sensitive to shortfall in financial gain, so lambda>1
rf=1.0303 #Risk free Rate

#****************************************************************************************************
# set the number of combinations for imaginary consumnption growth
randomdraws = 10000

#Scale factor so that utility of consumption remains comparable in magnitude to utility of recent gains or losses. If set to 0, investor doesn't care.
b0=np.arange(0,10.01,0.01) 

#****************************************************************************************************
# populate the empty lists with each epsilon, nu and consumption growth
for i in range(randomdraws):
    e1 = np.random.normal(0,1)
    e2 = e1*0.02
    epsilon.append(e2)
epsilon=np.array(epsilon)

g=np.exp(0.02+epsilon) #Consumption Growth
growth.append(g)

#****************************************************************************************************
#Error Term

def ErrorTerm(b0,g,x):
    R=pd.DataFrame(x*g)
    v=pd.DataFrame(np.ones(len(g)))
    v[R>=1.0303]=R-1.0303
    v[R<1.0303]=2*(R-1.0303)
    error_x=0.99*b0*(v.mean(axis=0).values)+0.99*x-1
    return error_x

#****************************************************************************************************
#Iterative Procedure - Bisection Search
    
x_data=pd.DataFrame([],columns=['EquilibriumX','Scale Factor'])

for i in b0:
    x=[1.0, 1.1] #Set x-=1 & x+=1.1.
    error_x0=ErrorTerm(i,g,x[0])
    error_x1=ErrorTerm(i,g,x[1])
    while error_x0<0 and error_x1>0: #Confirm that e(x-)<0 and e(x+)>0
        x_i=0.5*(x[0]+x[1])
        error_xi=ErrorTerm(i,g,x_i)
        if abs(error_xi)<10**-4:
            x_data.loc[i,:]=[x_i,i] #test[i]=pd.Series(x_i)
            break
        elif error_xi<0:
            x[0]=deepcopy(x_i)
        else:
            x[1]=deepcopy(x_i)
    else:
         print('''EquilibriumX initialization window proximity error for 
              b0 = %0.2f. Reinitialize with new window.''' %(i))
#****************************************************************************************************
# Price-dividend ratio 
x_data=x_data.assign(Price_Dividend=lambda x:(1/(x['EquilibriumX']-1)))

#Plot
fig1=plt.figure(figsize=(7,6))
plt.plot(x_data['Scale Factor'].values,x_data['Price_Dividend'].values)
plt.title('Price-dividend vs Scale Factor')
plt.xlabel('Scale Factor, b0')
plt.ylabel('Price-dividend ratio')

#****************************************************************************************************
# Expected market return and Equity premium 
ExpectedMktReturn=[np.average(i*g) for i in x_data['EquilibriumX'].values]
Equity_premium=pd.DataFrame(ExpectedMktReturn,columns=['Expected Market Return'])
Equity_premium=Equity_premium.assign(Equity_premium=lambda x:(x['Expected Market Return']-rf))

#Plot
fig2=plt.figure(figsize=(7,6))
plt.plot(x_data['Scale Factor'].values,Equity_premium['Equity_premium'].values)
plt.title('Equity Premium vs Scale Factor')
plt.xlabel('Scale Factor, b0')
plt.ylabel('Equity Pemium (E[Rm]-Rf)')

#****************************************************************************************************












