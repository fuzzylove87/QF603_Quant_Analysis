clear;
clc;
%% *How to ...*

%% HT001: Use one command to import data from a CSV file, |HT001a.csv| , using the first row as column labels, and the first column as row labels. Name the data imported as |data|.

% slide 169
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)

%% 
%
% *What if the file is |newfolder\newfile.csv| ?*
%

% slide 171 (Windows)
data=readtable('newfolder\newfile.csv')

% slide 171 (Mac)
% data=readtable('newfolder\newfile.csv')

%% 
%
% *What if the labels are as in |HT001b.csv| ?*
%

% slide 173
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true)

%% 
%
% *column labels*
%

% slide 174 (1)
data.Properties.VariableNames

%% 
%
% *row labels*
%

% slide 174 (2)
data.Properties.RowNames

%% HT002: Use an appropriate position-based indexing/slicing method to select data in data.
%% 
%
% *position-based indexing*
%

% slide 176 (1)
data{1,1}

%%
%

% slide 176 (2)
data(1,1)

%% 
%
% *position-based slicing (1)*
% 

% slide 178 (1)
data{1:2, 1:2}

%%
%

% slide 178 (2)
data(1:2, 1:2)

%% 
%
% *position-based slicing (2): the 1st row*
%

% slide 180 (1)
data{1, :}

%%
%

% slide 180 (2)
data(1, :)

%% 
%
% *position-based slicing (3): the 1st column*
%

% slide 182 (1)
data{:, 1}

%%
%

% slide 182 (2)
data(:, 1)

%% 
%
% *|data{:,:}|*
%

% slide 184 (1)
data{:,:}

%%
%
% *|data(:,:)|*
%

% slide 184 (2)
data(:,:)

%% 
%
% *fancy indexing*
%

% slide 186 (1)
data{[3 1],[1 2]}

%%
%

% slide 186 (2)
data([3 1],[1 2])

%% HT003: Use an appropriate label-based indexing/slicing method to select data in |data|.

%% 
%
% *label-based indexing*
%

% slide 188 (1)
data{'2','b'}

%%
%

% slide 188 (2)
data('2','b')

%% 
%
% *label-based slicing (1)*
%

% slide 189
% Oops! NO SOLUTION

%% 
%
% *label-based slicing (2): row with label |2|*
%

% slide 191 (1)
data{'2',:}

%%
%

% slide 191 (2)
data('2',:)

%%
%
% *label-based slicing (3): column with label |a|*
%

% slide 193 (1)
data{:, 'a'}

%%
%

% slide 193 (2)
data(:, 'a')

%% 
%
% *label-based fancy indexing*
%

% slide 195 (1)
data{{'2','3'},{'b','x1'}}

%%
%

% slide 195 (2)
data({'2','3'},{'b','x1'})

%% HT004: Add a column to |data|

%% 
%
% *add a column (1): a number*
%

%%
% *Error: |data.f1=1|*
%
% To assign to or create a variable in a table, 
% the number of rows must match the height of the table.

% slide 202 (1)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
%data.f1=1 %Wrong

%%
%
% *|data.f1(:)=1|*
%
% No error. But it is not what we want.
%

% slide 202 (2)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data.f1(:)=1

%%
%
% *|data1.f(:,1)=1|*
%

% slide 202 (3)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data.f1(:,1)=1

%%
%
% *|data{:,'f1'}=1|*
%

% slide 202 (4)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data{:,'f1'}=1

%%
%
% *Error: |data(:,'f1')=1|*
% 
% Right hand side of an assignment into a table must be another 
% table or a cell array.
%

% slide 202 (5)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
%data(:,'f1')=1 %Wrong

%%
%
% *|data(:,'f1')={1}|*
%

% slide 202 (6)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data(:,'f1')={1}

%% 
%
% *Add a column (2): an array*
%

%%
%
% *Error: |data.f1=[1,2,3,4]|*
%
% To assign to or create a variable in a table, 
% the number of rows must match the height of the table.
%

% slide 203 (1)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
%data.f1=[1,2,3,4]

%%
%
% *|data.f1=[1;2;3;4]|*
%

% slide 203 (2)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data.f1=[1;2;3;4]

%%
%
% *|data.f1={1;2;3;4}|*
%
% No error. But it is not what we want.
%

% slide 203 (3)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data.f1={1;2;3;4}

%%
%
% *Error: |data.f1(:)=[1;2;3;4]|*
%
% In an assignment  A(:) = B, the number of elements in A and B 
% must be the same.
%

% slide 203 (4)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
%data1.f1(:)=[1;2;3;4] #Wrong

%%
%
% *Error: |data.f1(:)={1;2;3;4}|*
%
% In an assignment  A(:) = B, the number of elements in A and B 
% must be the same.
%

% slide 203 (5)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
%data.f1(:)={1;2;3;4} #Wrong

%%
%
% *Error: |data.f1(:)=1:4|*
%
% In an assignment  A(:) = B, the number of elements in A and B 
% must be the same.
%

% slide 203 (6)
%data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
%data.f1(:)=1:4 #Wrong

%%
%
% *|data.f1(:,1)=[1;2;3;4]|*
%

% slide 203 (7)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data.f1(:,1)=[1;2;3;4]

%%
%
% *|data.f1(:,1)=1:4|*
%

% slide 203 (8)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data.f1(:,1)=1:4

%%
%
% *|data{:,'f1'}=[1;2;3;4]|*
%

% slide 203 (9)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data{:,'f1'}=[1;2;3;4]

%%
%
% *|data{:,'f1'}=1:4|*
%
% No error. But it is not what we want.
%

% slide 203 (10)
data=readtable('HT001b.csv','ReadRowNames',true,'ReadVariableNames',true);
data{:,'f1'}=1:4

%%
%
% *Homework: Use one command to add a column to |data|, 
% using the first row of data, and name this column |f|.*
%

% slide 204
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true);


%% HT005: Add a row to |data|

%%
%
% *|data{'4',:}=[1,2,3,4]|*
%

% slide 215
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
%data{'4',:}=1
%data('4',:)={1}
%data{'4',:}=[1,2,3,4]
data{'4',:}=1:4

%%
%
% *Error: |data{'4',:}=[1;2;3;4]|*
%
% The value being assigned from should have 4 columns. 
%

%data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
%data{'4',:}=[1;2;3;4] %Wrong

%% HT006: Delete a row from |data|

%%
%
% *Delete a column using the dot syntax*
%

% slide 221 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
% delete column b
data.b=[]

%%
%
% *Delete a column using label-based indexing*
%

% slide 221 (2)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
% delete column b
data(:,'b')=[]

%%
%
% *Delete a column using position-based indexing*
%

% slide 221 (3)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
% delete the first column
data(:,1)=[]

%%
%
% *Delete a row by the row number*
%

% slide 222 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
% delete the first row
data(1,:)=[]

%%
%
% *Delete a row by the row name*
%

% slide 222 (2)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
% delete row '0'
data('0',:)=[]

%% How to swap two values in variables |a| and |b|?


%% 
% 

%slide 226 (1)
a=1;
b=2;
temp=a;
a=b;
b=temp;
a
b

%%
% 

%slide 226 (2)
a=1;
b=2;
a, b=a, a  %No error.

%%
% 

%slide 226 (3)
a=1;
b=2;
[a, b]=deal(b,a)

%% HT007: Swap two rows/columns in |data|
%
% *Swap two columns, label-based (1)*

% slide 246 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data(:,{'b','c'})=data(:,{'c','b'})


%%
%
% *Swap two columns, label-based (2)*

% slide 246 (2)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data{:,{'b','c'}}=data{:,{'c','b'}}

%%
%
% *Swap two rows, label-based (1)*

% slide 248 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data({'0','1'},:)=data({'1','0'},:)

%%
%
% *Swap two rows, label-based (2)*

% slide 248 (2)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data{{'0','1'},:}=data{{'1','0'},:}

%%
%
% *Swap two columns, position-based (1)*

% slide 249 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data(:,[1,2])=data(:,[2,1])

%%
%
% *Swap two columns, position-based (2)*

% slide 249 (2)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data{:,[1,2]}=data{:,[2,1]}

%%
%
% *Swap two rows, position-based (1)*

% slide 249 (3)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data([1,2],:)=data([2,1],:)

%%
%
% *Swap two rows, position-based (2)*

% slide 249 (4)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data{[1,2],:}=data{[2,1],:}

%%
%
% *Use temp*
%

% slide 250 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
temp=data(1,:);
data(1,:)=data(2,:);
data(2,:)=temp

%%
%
% *Use deal*
%

% slide 250 (2)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
[data(1,:),data(2,:)]=deal(data(2,:), data(1,:))


%% HT008: Apply a function to each row/column of |data|
%
% *|rowfun| (1)*
%

% slide 262
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
rowfun(@(x1,x2,x3,x4) x1+x2+x3+x4, data)
data(:,'row_sum')=rowfun(@(x1,x2,x3,x4) x1+x2+x3+x4, data)

%%
%
% *|rowfun| (2)*
%

% slide 263
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
rowfun(@(x1,x2,x3,x4) x1+x2+x3+x4, data, 'OutputFormat', 'uniform')
data{:,'row_sum'}=rowfun(@(x1,x2,x3,x4) x1+x2+x3+x4, data, ...
                         'OutputFormat', 'uniform')

%%
%
% *|varfun| (1)*
%

% slide 264
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
varfun(@(x) sum(x), data)
data('column_mean',:)=varfun(@(x) sum(x), data)


%%
%
% *|varfun| (2)*
%

% slide 265
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
varfun(@(x) sum(x), data, 'OutputFormat', 'uniform')
data{'column_mean',:}=varfun(@(x) sum(x), data, 'OutputFormat', 'uniform')

%% HT009: Basic operations on two rows/columns of |data|
%

%%
%
% *Update |data| by adding the first row to every row in |data|*
%

% slide 284
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data{:,:}=data{:,:}+repmat(data{1,:},size(data,1),1)

%%
%
% *Add the first row and the last column (without finding the size of
% |data|) element wise and store the result in a column array.*
%

% slide 285
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
transpose(data{1,:})+data{:,end}

%%
%
% *Add elements in the first column from the second row to the last row 
% (without finding the size of |data|) and elements in the second column 
% from the first row to the second to the last row (without finding the 
% size of |data|) element wise and store the result in a row array.*
%

% slide 286
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
transpose(data{2:end,1}+data{1:end-1,2})

%%
%
% *Use one command to compute the matrix multiplication, 
% using the first 3 rows in |data| as the first matrix, 
% and using the last row (without finding the size of |data|) 
% as the second 1-column matrix.
%

% slide 288
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)
data{1:3, :}*transpose(data{end,:})

%% HT010: Sort |data|
%

%%
%
% *|sort| a matrix*
%

% slide 295 (1)
x=[1 2 3;3 1 2;2 1 3]

%%
%
% *|[y,I]=sort(x,1,'ascend')|*
%

% slide 295 (2)
[y,I]=sort(x,1,'ascend')

%%
%
% *|x(I(:,2),:)| sort rows of |x| using the second column*
%

% slide 295 (3)
x(I(:,2),:)

%%
%
% *|[y,I]=sort(x,2,'ascend')|*
%

% slide 295 (4)
[y,I]=sort(x,2,'ascend')

%%
%
% *|x(:,I(:,2))| sort columns of|x| using the second row*
%

% slide 295 (5)
x(:,I(2,:))

%%
%
% *|sort| |data|*
%

% slide 296 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)

%%
%

% slide 296 (2)
[y,I]=sort(data{:,:},1,'ascend')

%%
%
% *|sort| rows of |data| using the 3rd column*
%

% slide 296 (3)
data(I(:,3),:)


%%
%

% slide 296 (4)
[y,I]=sort(data{:,:},2,'ascend')

%%
%
% *|sort| columns of |data| using the 3rd row*
%

% slide 296 (5)
data(:,I(3,:))

%%
% 

% slide 298 (1)
x=[1 3 2;2 1 3;1 2 3]

%%
%
% *|sortrows(x)|*
%

% slide 298 (2)
sortrows(x)

%%
%
% *|sortrows(x,[1,-2])|*
%

% slide 298 (3)
sortrows(x,[1,-2])


%%
%
% *Use |sortrows(x,2)| to sort |x| using the second column*
%

% slide 298 (4)
sortrows(x,2)

%%
%
% *Use |sortrows| to sort |x| using the second row*
%

% slide 299 
x=[1 3 2;2 1 3;1 2 3]
transpose(sortrows(transpose(x),2))

%%
%
% *Use |sortrows| to sort |data| using the 3nd column*
%

% slide 300 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)

%%
%

% slide 300 (2)
sortrows(data,3)

%%
%

% slide 300 (3)
data

%%
%

% slide 300 (4)
data=sortrows(data,3)


%%
%
% *Use |sortrows| to sort |data| using the 3nd row*
%

% slide 301 (1)
data=readtable('HT001a.csv','ReadRowNames',true,'ReadVariableNames',true)

%%
%

% slide 301 (2)
[y,I]=sortrows(transpose(data{:,:}),3);
data(:,I)

%% 
%
% *Indexing on Assignment*
%

% slide 302 (1)
x=[1 2 3;4 5 6;7 8 9]

%%
%

% slide 302 (2)
x(2,:)=x(:,3)