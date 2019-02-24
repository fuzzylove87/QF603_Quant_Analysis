import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.optimize import brentq
import matplotlib.pylab as plt
import datetime as dt
from scipy import interpolate
from scipy.optimize import least_squares
from math import exp


# Changes
# DD optimized against market IV curve for OTM calls and puts 


# =============================================================================
# START OF PART 2
# =============================================================================

rate_df = pd.read_csv('discount.csv')
interp = interpolate.interp1d(rate_df.iloc[:,0],rate_df.iloc[:,1])

today = dt.date(2013, 8, 30)
expiry = dt.date(2015, 1, 17)
D = (expiry-today).days
T = D/365.0
S = 846.9
r = interp(D)/100
F = S*exp(r*T)


def fetch():
    googcall = pd.read_csv('goog_call.csv')
    googcall['call'] = (googcall['best_bid']+googcall['best_offer'])/2
    googput = pd.read_csv('goog_put.csv')
    googput['put'] = (googput['best_bid']+googput['best_offer'])/2
    global goog 
    goog = pd.concat([googcall['strike'],googcall['call'],googput['put']], axis=1)
    return goog

fetch()

goog['callonly']= goog.apply(lambda row: row['call'] if row['strike']>F else np.nan,axis=1)
goog['putonly']= goog.apply(lambda row: row['put'] if row['strike']<F else np.nan,axis=1)
strikes = goog['strike']
strikes_c = [i for i in goog['strike'] if i>F]
strikes_p = [i for i in goog['strike'] if i<F]
call_price = [i for i in goog['callonly'] if i>0]
put_price = [i for i in goog['putonly'] if i>0]
realprices = put_price + call_price 





# =============================================================================
# Extracted ATM SIGMA using Black Scholes
# =============================================================================


sigma = (0.25425618293224583 + 0.26229211463233354)/2


# =============================================================================
# 
# =============================================================================




def DD_Call (S, K, r, sigma, T, Beta):
    if Beta==0:
        F=(S*np.exp(r*T))
        x_Star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*((F-K)*ss.norm.cdf(-x_Star,0,1)+
                             F*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))
    else:
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        d1 = (np.log(F/K)+(sigma**2)/2*T)/(sigma*np.sqrt(T))
        d2 = (np.log(F/K)-(sigma**2)/2*T)/(sigma*np.sqrt(T))
        return np.exp(-r*T)*(F*ss.norm.cdf(d1,0,1)-K*ss.norm.cdf(d2,0,1))

def DD_Put (S, K, r, sigma, T, Beta):
    if Beta==0:
        F=(S*np.exp(r*T))
        x_Star = (K-F)/(F*sigma*np.sqrt(T))
        return np.exp(-r*T)*((K-F)*ss.norm.cdf(x_Star,0,1)+
                             F*sigma*np.sqrt(T)*ss.norm.pdf(-x_Star,0,1))
    else:
        F=(S*np.exp(r*T))/Beta
        K=K+((1-Beta))*F
        sigma=Beta*sigma
        d1 = (np.log(F/K)+(sigma**2)/2*T)/(sigma*np.sqrt(T))
        d2 = (np.log(F/K)-(sigma**2)/2*T)/(sigma*np.sqrt(T))
        return np.exp(-r*T)*(K*ss.norm.cdf(-d2,0,1)-F*ss.norm.cdf(-d1,0,1))
    
    
    



# =============================================================================
# 
# =============================================================================
    



def impliedCallVolatility(S, K, r, price, T,Beta):
    impliedVol = brentq(lambda x: price -
                        DD_Call(S, K, r, x, T, Beta),
                        1e-6, 1)

    return impliedVol

def impliedPutVolatility(S, K, r, price, T, Beta):
    impliedVol = brentq(lambda x: price -
                        DD_Put(S, K, r, x, T, Beta),
                        1e-6, 1)

    return impliedVol






    
# =============================================================================
# DD IV generation
# =============================================================================



DD_IV = pd.DataFrame(strikes.values, columns=['Strikes'])
   


def Theo_Pricing(Beta):
    Theo_Price=[]
    for K in strikes:
        Theo_Price.append([K,DD_Put(S, K, r, sigma, T, Beta),DD_Call(S, K, r, sigma, T, Beta)])
        
    Theo_Price = pd.DataFrame(Theo_Price, columns=['strike','put','call'])
    Theo_Price['callonly']= Theo_Price.apply(lambda row: row['call'] if row['strike']>F else np.nan,axis=1)
    Theo_Price['putonly']= Theo_Price.apply(lambda row: row['put'] if row['strike']<F else np.nan,axis=1)
    call_theo = [i for i in Theo_Price['callonly'] if i>0]
    put_theo = [i for i in Theo_Price['putonly'] if i>0]

    
    summary=[]
    for K,i in zip(strikes_p,range(len(strikes_p))):
        price = put_theo[i]
        impliedvol = impliedPutVolatility(S, K, r, price, T,Beta=1)
        summary.append(impliedvol)
        
    for K,i in zip(strikes_c,range(len(strikes_c))):
        price = call_theo[i]
        impliedvol = impliedCallVolatility(S, K, r, price, T,Beta=1)
        summary.append(impliedvol)
    
    DD_IV['DD IV - Beta '+str(Beta)] = pd.Series(summary)
     

Theo_Pricing(0.0)
Theo_Pricing(0.2)
Theo_Pricing(0.4)
Theo_Pricing(0.6)
Theo_Pricing(0.8)
Theo_Pricing(1.0)



# =============================================================================
# Get Market impvol using DD Beta 1
# =============================================================================


marketiv = []

def GetIV_DD ():

    for K,i in zip(strikes_p,range(len(strikes_p))):
        price = put_price[i]
        impliedvol = impliedPutVolatility(S, K, r, price, T,Beta=1)
        marketiv.append(impliedvol)
        
    for K,i in zip(strikes_c,range(len(strikes_c))):
        price = call_price[i]
        impliedvol = impliedCallVolatility(S, K, r, price, T,Beta=1)
        marketiv.append(impliedvol)


GetIV_DD()


marketiv_p = marketiv[:65].copy() #65
marketiv_c = marketiv[65:].copy() #57



# =============================================================================
# DD calibration - match option prices for calls and puts
# =============================================================================


df2 = pd.DataFrame(goog[['strike','call','put']])
df2['real']=realprices
df2.set_index(df2.columns[0], inplace=True)





    
def DDIV_Call (S, K, r, sigma, T,Beta):
    ddiv = brentq(lambda x: DD_Call(S, K, r, sigma, T, Beta) -
                        DD_Call(S, K, r, x, T, Beta=1),
                        1e-6, 1)
    return ddiv

def DDIV_Put (S, K, r, sigma, T, Beta):
    ddiv = brentq(lambda x: DD_Put(S, K, r, sigma, T, Beta) -
                        DD_Put(S, K, r, x, T, Beta=1),
                        1e-6, 1)
    return ddiv



def DD_Calibrate_All (x):
    err = 0.0
    for i, p in zip(range(len(strikes_p)),marketiv_p):
        err += (p - DDIV_Put(S, strikes_p[i], r, sigma, T, x[0]))**2
    for i, p in zip(range(len(strikes_c)),marketiv_c):
        err += (p - DDIV_Call(S, strikes_c[i], r, sigma, T, x[0]))**2
    return err



initialGuess = 0.5

res_all = least_squares(lambda x: DD_Calibrate_All(x),initialGuess)


print(res_all.x[0])


Theo_Pricing(res_all.x[0])


DD_IV.rename(columns = {DD_IV.columns[-1]:'DD Calibrated'},inplace =True)




# =============================================================================
# 
# =============================================================================






df = DD_IV.copy()
df.set_index('Strikes', inplace=True)
colours = ['#303F9F','#59309F','#921CA5','#A51C74','#A51C2F','#A54D1C']

fig1, ax1 = plt.subplots(tight_layout=True)
for b,c in zip(range(len(df.columns)),colours):
    fig1 = plt.plot(df.iloc[:,b], c, linewidth=1.5, linestyle=':', markersize=3)

df['Market IV'] = marketiv
ax1 = plt.plot(df['Market IV'], '#00B10B', marker='s', linestyle='', markersize=5)



ax1 = plt.plot(df['DD Calibrated'], 'b', linewidth=1.8, linestyle='-', markersize=3)
               
ax1 = plt.legend()
ax1 = plt.ylabel('Implied Lognormal Volatility')
ax1 = plt.xlabel('Strikes')
ax1 = plt.grid(linestyle='--')
ax1 = plt.title('Displaced Diffusion Model Calibration')
#ax1 = plt.savefig('DD.png', dpi=300,quality=100)


# =============================================================================
# 
# =============================================================================




def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if F == K:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom
    return sabrsigma



sabr1=[]
for i in strikes:
    sabr1.append(SABR(F,i,T, 1, 0.8, -0.3, 0.3))


df['SABR Uncalibrated'] = sabr1

fig2, ax2 = plt.subplots(tight_layout=True)

ax2 = plt.plot(df['Market IV'], '#00B10B', marker='s', linestyle='', markersize=5)

               
ax2 = plt.legend()
ax2 = plt.ylabel('Implied Lognormal Volatility')
ax2 = plt.xlabel('Strikes')
ax2 = plt.grid(linestyle='--')
ax2 = plt.title('SABR Model Calibration')


# =============================================================================
# SABR calibration - match option impvol as implied by BS model
# =============================================================================



def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], 0.8, x[1], x[2]))**2

    return err


initialGuess = [0.02, 0.2, 0.1]

res = least_squares(lambda x: sabrcalibration(x,
                                              strikes.values,
                                              df['Market IV'].values,
                                              F,
                                              T),
                    initialGuess)
alpha = res.x[0]
beta = 0.8
rho = res.x[1]
nu = res.x[2]

print(alpha,beta,rho,nu)


sabr2=[]
for i in strikes:
    sabr2.append(SABR(F,i,T, alpha, 0.8, rho, nu))



df['SABR Calibrated'] = sabr2
plt.plot(df['SABR Calibrated'], '#FFB200', linewidth=2)
plt.legend()
#ax2 = plt.savefig('SABR.png', dpi=300,quality=100)

# =============================================================================
# 
# =============================================================================




alpha_test = pd.DataFrame(strikes.values)
alpha_list = [0.6,0.8,1.0,1.2,1.4]
for alp in alpha_list:
    sabr_alpha = []
    for i in strikes:
        sabr_alpha.append(SABR(F,i,T, alp, beta, rho, nu))
    alpha_test['SABR Alpha '+str(alp)] = pd.Series(sabr_alpha)
alpha_test.set_index(alpha_test.columns[0], inplace=True)
alpha_test['SABR Calibrated'] = sabr2



fig4, ax1 = plt.subplots(1,3, figsize=(12, 4), tight_layout=True )


ax1 = plt.subplot(1,3,1)
colours2 = ['#95F838','#41E94E','#4ADA55','#53CB5C','#5CBC63','k']
for b,c in zip(range(len(alpha_test.columns)),colours2):
    ax1 = plt.plot(alpha_test.iloc[:,b], c, linewidth=2, linestyle='-')          
ax1 = plt.legend()

    
 
ax1 = plt.ylabel('Implied Lognormal Volatility')
ax1 = plt.xlabel('Strikes')
ax1 = plt.grid(linestyle='--')
ax1 = plt.title('SABR Model - Varying Alpha')




# =============================================================================
# 
# =============================================================================
               




rho_test = pd.DataFrame(strikes.values)
rho_list = [-0.9,-0.6,-0.3,0,0.3]
for rh in rho_list:
    sabr_rho = []
    for i in strikes:
        sabr_rho.append(SABR(F,i,T, alpha, beta, rh, nu))
    rho_test['SABR Rho '+str(rh)] = pd.Series(sabr_rho)
rho_test.set_index(rho_test.columns[0], inplace=True)
rho_test['SABR Calibrated'] = sabr2


ax3 = plt.subplot(1,3,2)


colours2 = ['#319EC6','#3182C6','#3B6ABD','#3151C6','#3831C6','k']
for b,c in zip(range(len(rho_test.columns)),colours2):
    ax3 = plt.plot(rho_test.iloc[:,b], c, linewidth=2, linestyle='-')          
ax3 = plt.legend()

ax3 = plt.ylabel('Implied Lognormal Volatility')
ax3 = plt.xlabel('Strikes')
ax3 = plt.grid(linestyle='--')
ax3 = plt.title('SABR Model - Varying Rho')



# =============================================================================
# 
# =============================================================================







nu_test = pd.DataFrame(strikes.values)
nu_list = [-0.2,0.1,0.4,0.7,1]
for nut in nu_list:
    sabr_nu = []
    for i in strikes:
        sabr_nu.append(SABR(F,i,T, alpha, beta, rho, nut))
    nu_test['SABR Nu '+str(nut)] = pd.Series(sabr_nu)
nu_test.set_index(nu_test.columns[0], inplace=True)
nu_test['SABR Calibrated'] = sabr2


ax4 = plt.subplot(1,3,3)


colours2 = ['#BC5CB2','#CB53BE','#DA4ACB','#E941D7','#F838E4','k']
for b,c in zip(range(len(nu_test.columns)),colours2):
    ax4 = plt.plot(nu_test.iloc[:,b], c, linewidth=2, linestyle='-')          
ax4 = plt.legend()

    
 
ax4 = plt.ylabel('Implied Lognormal Volatility')
ax4 = plt.xlabel('Strikes')
ax4 = plt.grid(linestyle='--')
ax4 = plt.title('SABR Model - Varying Nu')

#ax4 = plt.savefig('SABR-All.png', dpi=300,quality=100)


# =============================================================================
# END OF PART 2
# =============================================================================
