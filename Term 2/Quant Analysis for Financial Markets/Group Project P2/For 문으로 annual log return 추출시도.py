# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:43:18 2018

@author: ykfri
"""


# Extract log return for each index
DJI_Last_Price = []
DJI_Annual_Logreturn = []
GSPC_Last_Price = []
GSPC_Annual_Logreturn = []

if Last_Annual_BD == []:
    for i in Last_Monthly_BD:
        if i.month == 12:
            Last_Annual_BD.append(i)

if DJI_Last_Price == []:
    for i in Last_Annual_BD:
        for k in DJI['Date']:
            if i == k:
                DJI_Last_Price.append(DJI['Close'].loc[k])

if GSPC_Last_Price ==[]:
    for i in Last_Annual_BD:    
        for k in GSPC['Date']:
            if i == k:
                GSPC_Last_Price.append(GSPC['Close'].loc[k])

for i in range(len(DJI_Last_Price)-1):
    DJI_Annual_Logreturn.append(DJI_Last_Price[i+1]/DJI_Last_Price[i])
    
for i in range(len(GSPC_Last_Price)-1):
    GSPC_Annual_Logreturn.append(GSPC_Last_Price[i+1]/GSPC_Last_Price[i])

DJI_Annual_Logreturn = np.array(DJI_Annual_Logreturn)
GSPC_Annual_Logreturn = np.array(GSPC_Annual_Logreturn)
