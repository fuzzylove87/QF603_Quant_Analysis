classdef Call_option
    properties
        S0;
        K;
        T;
        r;
        sigma;
    end
    methods
        function obj=call_option(obj, S0, K, T, r, sigma)
            obj.S0 = float(S0);
            obj.K = K;
            obj.T = T;
            obj.r = r;
            obj.sigma = sigma;
        end
        function value(obj)
            d1 = ((log(obj.S0 / obj.K) + (obj.r+ 0.5 * obj.sigma^ 2) * obj.T)/(obj.sigma* sqrt(obj.T)))
            d2 = ((log(obj.S0 / obj.K) + (obj.r-0.5 * obj.sigma^ 2) * obj.T)/(obj.sigma* sqrt(obj.T)))
            value = (obj.S0 * normcdf(d1, 0.0, 1.0)-obj.K* exp(-obj.r* obj.T) * normcdf(d2, 0.0, 1.0))
            return value
        end
        function vega(obj):
            d1 = ((log(obj.S0 / obj.K) + (obj.r+ 0.5 * obj.sigma^ 2) * obj.T)/(obj.sigma* sqrt(obj.T)))
            vega= obj.S0 * normcdf(d1, 0.0, 1.0) * sqrt(obj.T)
            return vega
        function imp_vol(obj, C0, sigma_est=0.2, it=100):
            option = call_option(obj.S0, obj.K, obj.T, obj.r, sigma_est)
            for i = range(it):
                option.sigma-= (option.value() -C0) / option.vega()
            return option.sigma