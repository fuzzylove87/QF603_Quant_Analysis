classdef Calloption
    properties
        S0;
        K;
        T;
        r;
        sigma;
    end
    methods
        function obj=Calloption(S0, K, T, r, sigma)
            obj.S0 = S0;
            obj.K = K;
            obj.T = T;
            obj.r = r;
            obj.sigma = sigma;
        end
        function [value2]=value(obj)
            d1 = ((log(obj.S0 / obj.K) + (obj.r+ 0.5 * obj.sigma^ 2) * obj.T)/(obj.sigma* sqrt(obj.T)));
            d2 = ((log(obj.S0 / obj.K) + (obj.r-0.5 * obj.sigma^ 2) * obj.T)/(obj.sigma* sqrt(obj.T)));
            value2 = (obj.S0 * normcdf(d1, 0.0, 1.0)-obj.K* exp(-obj.r* obj.T) * normcdf(d2, 0.0, 1.0));
        end
        function [vega2]=vega(obj)
            d1 = ((log(obj.S0 / obj.K) + (obj.r+ 0.5 * obj.sigma^ 2) * obj.T)/(obj.sigma* sqrt(obj.T)));
            vega2= obj.S0* normcdf(d1, 0.0, 1.0) * sqrt(obj.T);
        end    
        function [os]=imp_vol(obj, C0, sigma_est, it)
            if nargin==3
                it=100;
            elseif nargin==2
                sigma_est=0.2;
                it=100;
            end
            option = Calloption(obj.S0, obj.K, obj.T, obj.r, sigma_est);
            for i = range(it)
                option.sigma = option.sigma-(option.value()-C0) / option.vega();
                os=option.sigma;
            end
            end
        end
end