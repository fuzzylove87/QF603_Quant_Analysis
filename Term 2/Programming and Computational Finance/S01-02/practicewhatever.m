classdef A
    properties
        values
    end
    methods
       function obj=A(val)
            obj.value=val;
       end
       function r=roundoff(obj)
           r=round(obj.value,2);
       end
    end
end
    
    