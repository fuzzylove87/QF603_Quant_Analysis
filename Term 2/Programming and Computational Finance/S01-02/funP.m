function [P]=funP(PV, r, t)
P=(r/12*PV)/(1-(1+r/12)^(-12*t));
end