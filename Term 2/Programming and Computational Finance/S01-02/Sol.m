function [a] = same_row(i,j)
    a = floor((i)/9)==floor((j)/9);
end
function [a] = same_col(i, j)
    a = mod(i, 9)==mod(j, 9);
end
function [a] =same_block(i, j)
    a = floor(i/27)==(j/27) && floor(mod(i,9)/3)==floor(mod(j,9)/3);
end
function [r]=Sudoku(s)
    c=strfind(s,'0');
    if isempty(c)
        disp(s);
    else
        i = c(1)-1;
        excluded_numbers =[];
        for j = 1:81
            if same_row(i,j-1)|| same_col(i-1,j-1)|| same_block(i,j-1)
                excluded_numbers = unique([excluded_numbers, s(j)]);
            end
        end
        numbers = setdiff(['123456789'], excluded_numbers);
        for m = numbers
            r([s(1:i), m, s(i+2:81)])
        end
    end
end