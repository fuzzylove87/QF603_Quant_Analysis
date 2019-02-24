function f=flist2()
   fun_list={};
   for i=[3, 5]
      fun_list=[fun_list, {@(x) i*x}];
   end
   f=fun_list;
end