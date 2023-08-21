# MIT License

# Copyright (c) 2023, Technical University of Denmark (DTU)

#

# Permission is hereby granted, free of charge, to any person obtaining a copy

# of this software and associated documentation files (the "Software"), to deal

# in the Software without restriction, including without limitation the rights

# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell

# copies of the Software, and to permit persons to whom the Software is

# furnished to do so, subject to the following conditions:

#

# The above copyright notice and this permission notice shall be included in all

# copies or substantial portions of the Software.
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numba

@numba.njit
def sys2(t, y,k=1, mu1=1, mu2=1, Ks1=1, Ks2=1, YXs1=1, YXs2=1, ms1=1, ms2=1, G=0, M=0 ):
    y1, y2, y3, y4 = y

    q1 = mu1 * (y1 / (y1 + Ks1)) - ms1 * YXs1
    #q2 = mu2 * y2 / (y2 + (Ks2 / Ks1 * y1)) - ms2 * YXs2
    q2 = mu2 * y2 / (y2 + Ks2) - ms2 * YXs2
    qS = q1 + q2
    Yos = ((-4+YXs1*4.074)/-4)*(32/26.13)
    #Yos = (((YXs1+YXs2)*4.074)-7.33/-4)*(32/26.13)
    #Yos = 0.481
    dydt = np.empty(4)
    dydt[0] = -q1 * y3
    dydt[1] = -q2 * y3
    dydt[2] = (YXs1 * q1 + YXs2 * q2) * y3
    dydt[3] = k * (0.0083 - y4) - (Yos * qS * y3)
    #dydt[3] = 65 - (0.481 * qS * y3) + D * (0.0083 - y4)
    return dydt
dfX = pd.read_excel('BioL data 3.xlsx', sheet_name='Biomass')
dfO= pd.read_excel('BioL data 3.xlsx', sheet_name='DO')
Gluc=pd.read_excel('BioL data 3.xlsx', sheet_name='Glucose').T
Gluc=Gluc[0].tolist()
Ara=pd.read_excel('BioL data 3.xlsx', sheet_name='Arabinose').T
Ara=Ara[0].tolist()
Time=pd.read_excel('BioL data 3.xlsx', sheet_name='Time')
Time=Time['t(h)'].tolist()

realpar = [300, 0.14, 0.074, 0.049, 0.0022, 0.094, 0.046, 0.576, 0.596, 0.545, 0.0043, 0.011, 0.0038]
errorlist=[]
bounds = [(25,600), (0.01, 0.4), (0.01, 0.4), (0.0001, 0.4), (0.0001, 0.4), (0.3, 0.6), (0.3, 0.6), (0.0001, 0.05), (0.0001, 0.05)]
max_iterations = 1500
options = {'maxiter': max_iterations}

def objective(x, x_num):
    global Time
    errorlist = [] # Move inside the function
    k, mu1, mu2, Ks1, Ks2, YXs1, YXs2, ms1, ms2 = x
    sumerrortot = 0
    G = Gluc[x_num-1]
    A = Ara[x_num-1]
    y0 = [G, A, dfX[f'X{x_num}'][0], 0.0083]
    args = (0.14, 0.074, 0.0022, 0.094, 0.576, 0.596, 0.0043, 0.011, G, A)
    sol = solve_ivp(sys2, (0, 50), y0, method='Radau', dense_output=True, args=(k, mu1, mu2, Ks1, Ks2, YXs1, YXs2, ms1, ms2))
    t_eval = Time
    y = sol.sol(t_eval)
    S1 = y[0].tolist()
    S2 = y[1].tolist()
    X = y[2].tolist()
    O2 = y[3].tolist()
    O2 = (np.array(O2) / 0.0083)
    time = sol.t.tolist()
    time=np.array(time)
    Time=np.array(Time)
    X=np.array(X)
    DCW = dfX[f'X{x_num}']
    pO = dfO[f'X{x_num}'] / 100
    e1=(DCW-X)**2
    e2=(pO-O2)**2
    error=e1+e2
    sumerror=sum(error)
    sumerrortot=sumerror+sumerrortot
    errorlist.append(sumerrortot)
    return sumerrortot, errorlist

all_error_lists = []
df_params = pd.DataFrame(columns=['k', 'mu1', 'mu2', 'Ks1', 'Ks2', 'YXs1', 'YXs2', 'ms1', 'ms2'])

for i in range(48):
    print(f"Running optimization {i+1} of 48")
    result = minimize(lambda x, x_num: objective(x, x_num)[0], [200,0.2,0.2,0.2,0.1,0.4,0.4,0.02,0.02],
                      args=(i+1), bounds=bounds, method='Nelder-Mead', options=options)
    df_params.loc[i] = result.x.tolist()
    _, errorlist = objective(result.x, i+1)
    all_error_lists.append(errorlist)

print(df_params)

# Convert all_error_lists into a DataFrame
df_errors = pd.DataFrame(all_error_lists)

print(df_errors)

# Export both DataFrames to Excel
df_params.to_excel('df_params exp 2 no inh.xlsx')
df_errors.to_excel('df_errors exp 2 no inh.xlsx')