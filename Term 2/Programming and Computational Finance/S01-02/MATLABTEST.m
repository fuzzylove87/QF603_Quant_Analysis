PV=800000;
t=25;
r=2.6/100;
P=(r/12*PV)/(1-(1+r/12)^(-12*t));
disp(P);
disp(1, 2)