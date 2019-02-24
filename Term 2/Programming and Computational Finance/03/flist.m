function f=flist(x,y)
   fun_list={@f1, @f2};
   f=fun_list{mod(x+y,2)+1};
end

function y=f1(x)
   y=2*x;
end

function y=f2(x)
   y=x^2;
end