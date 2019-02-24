function PythonPrint(varargin)
   outputstr='';
   sep=' ';
   for i=1:nargin
      outputstr=[outputstr, sep, num2str(varargin{i})];
   end
   disp(outputstr);