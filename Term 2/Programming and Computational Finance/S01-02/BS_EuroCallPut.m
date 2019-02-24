function [c,p] = BS_EuroCallPut(S,K,r,q,sigma,T,t)
d1=(log(S/K)+(r-q+sigma^2/2)*(T-t))/(sigma*sqrt(T-t));
d2=d1-sigma*sqrt(T-t);
c=S*exp(-q*(T-t))*normcdf(d1)-K*exp(-r*(T-t))*normcdf(d2);
p=K*exp(-r*(T-t))*normcdf(-d2)-S*exp(-q*(T-t))*normcdf(-d1);
end