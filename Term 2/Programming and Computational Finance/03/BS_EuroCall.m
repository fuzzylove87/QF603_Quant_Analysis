function c=BS_EuroCall(S,K,r,q,sigma,T)
d1=(log(S/K)+(r-q+sigma^2/2)*T)/(sigma*sqrt(T));
d2=d1-sigma*sqrt(T);
c=S*exp(-q*T)*normcdf(d1)-K*exp(-r*T)*normcdf(d2);
end