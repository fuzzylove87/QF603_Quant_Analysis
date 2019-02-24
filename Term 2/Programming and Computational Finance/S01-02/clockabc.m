year = 2017;
o={'+','-','*','/',''};
for o1 = o
    for o2 = o
        for o3 = o
            for o4 = o
                for o5 = o
                    for o6 = o
                        for o7 = o
                            for o8 = o
                                s=['1', o1{1}, ...
                                   '2', o2{1}, ...
                                   '3', o3{1}, ...
                                   '4', o4{1}, ...
                                   '5', o5{1}, ...
                                   '6', o6{1}, ...
                                   '7', o7{1}, ...
                                   '8', o8{1}, '9'];
                                if eval(s)==year
                                   disp([s, '=', num2str(year)])
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
