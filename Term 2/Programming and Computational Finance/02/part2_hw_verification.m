data=readtable('dataset01.csv','ReadVariableNames',true)

BS=rowfun(@option_bs,data)
data{:,'BS'}=BS{:,1}

data1=readtable('HT001a.csv', 'ReadRowNames', true, 'ReadVariableNames', true)
data1.f=data1{1,:}'

function [V]=option_bs(S, K, r,q,sigma,T,c)
d1=(log(S/K)+(r-q+sigma^2)*T)/(sigma*sqrt(T));
d2=d1-sigma*sqrt(T);
V=S*exp(-q*T)*normcdf(d1)-K*exp(-r*T)*normcdf(d2);
end
