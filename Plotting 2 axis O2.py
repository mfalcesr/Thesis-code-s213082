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
import numba
@numba.njit
def sys2(t, y,k=1, mu1=1, mu2=1, Ks1=1, Ks2=1, YXs1=1, YXs2=1, ms1=1, ms2=1, G=0, M=0 ):
    y1, y2, y3, y4 = y

    q1 = mu1 * (y1 / (y1 + Ks1)) - ms1 * YXs1
    q2 = mu2 * y2 / (y2 + (Ks2 / Ks1 * y1)) - ms2 * YXs2
    #q2 = mu2 * y2 / (y2 + Ks2) - ms2 * YXs2
    qS = q1 + q2
    Yos = ((-4+YXs1*4.074)/-4)*(32/26.13)
    #Yos = 0.481
    dydt = np.empty(4)
    dydt[0] = -q1 * y3
    dydt[1] = -q2 * y3
    dydt[2] = (YXs1 * q1 + YXs2 * q2) * y3
    dydt[3] = k * (0.0083 - y4) - (Yos * qS * y3)
    #dydt[3] = 65 - (0.481 * qS * y3) + D * (0.0083 - y4)
    return dydt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import time as t

dfX = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Biomass')
dfO = pd.read_excel('BioL data Gluc.xlsx', sheet_name='DO')
Gluc = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Glucose').T
Gluc = Gluc[0].tolist()
Ara = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Arabinose').T
Ara = Ara[0].tolist()
Time = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Time')
Time = Time['t(h)'].tolist()
sumerror = 0

x = [33]
for i in x:
    sample = 'X'
    num = str(i)
    samplenum = sample + num
    DCW = dfX[samplenum]
    max_index = np.argmax(DCW)
    #DCW[max_index:] = DCW[max_index]
    pO = dfO[samplenum]
    pO = pO/100
    G = Gluc[i]
    A = Ara[i]
    y0 = [G, A, DCW[0], 0.0083]
    t_span = (Time[0], 50)

    args1 = [497.3589703, 0.420367052, 0.084062344, 0.2, 0.162636562, 0.599999898, 0.304049926, 0.006667857, 0.001069403, 4.9e-05]

    sol = solve_ivp(sys2, t_span, y0, method='LSODA', args=args1)
    S1 = sol.y[0].tolist()
    S2 = sol.y[1].tolist()
    X = sol.y[2].tolist()
    O2 = sol.y[3].tolist()
    O2 = (np.array(O2) / 0.0083)*100
    time = sol.t.tolist()

    fig, ax1 = plt.subplots(figsize=(10, 8))

    ax1.plot(Time, DCW, label='X experimental', linestyle='--')
    ax1.plot(sol.t, X, label='X')
    #ax1.plot(sol.t, S1, label='Glucose')
    #ax1.plot(sol.t, S2, label='Arabinose')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Biomass DW (g/L)', color='b')
    ax1.tick_params('y', colors='b')


    ax2 = ax1.twinx()
    ax2.plot(Time, pO * 100, label='pO2 experimental', linestyle='--', color='r')
    ax2.plot(sol.t, O2, label='pO2', color='g')
    ax2.set_ylabel('Oxygen %', color='r')
    ax2.tick_params('y', colors='r')

    # Merging legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('30 g/L Glucose')
    plt.xlim(0, 40)
    ax1.set_ylim(0, 40)
    ax2.set_ylim(0, 150)
    plt.show()
