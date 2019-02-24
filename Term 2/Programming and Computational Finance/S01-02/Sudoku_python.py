# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:18:33 2018

@author: ykfri
"""
'''
def same_row(i,j): return (i//9)==(j//9)
def same_col(i,j): return (i%9)==(j%9)
def same_block(i,j): return ((i//27)==(j//27)) and (((i%9)//3)==((j%9)//3))
def r(s):
    i=s.find('0')
    if i==-1:
        print(s)
    else:
        excluded_numbers={s[j] for j in range(81) if same_row(i,j)
                                                or same_col(i,j)
                                                or same_block(i,j)}
        for m in set('123456789')-excluded_numbers:
            r(s[:i]+m+s[i+1:])

s=('390060807' + '020030050' + '000005096' +
'900502400' + '000000000' + '003907002' +
'810600000' + '030050080' + '502090043')

print(s)
print(r(s))

'''

s=('390060807' + '020030050' + '000005096' +
'900502400' + '000000000' + '003907002' +
'810600000' + '030050080' + '502090043')

def same_row(i,j): return (i//9)==(j//9)
def same_col(i,j): return (i%9)==(j%9)
def same_block(i,j): return ((i//27)==(j//27)) and (((i%9)//3)==((j%9)//3))

i=s.find('0')
if i==-1:
    print(s)
else:
    excluded_numbers={s[j] for j in range(81) if same_row(i,j) or same_col(i,j) or same_block(i,j)}

print(excluded_numbers)