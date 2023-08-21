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
@numba.njit
def sys3(t, y, k=1, mu1=1, Ks1=1, YXs1=1, ms1=1, G=0):
    y1, y3, y4 = y
    q1 = mu1 * ((y1 / (y1 + Ks1)) - ms1 * YXs1)
    Yos = ((-4+YXs1*4.074)/-4)*(32/26.13)
    dydt = np.empty(3)
    dydt[0] = -q1 * y3
    dydt[1] = YXs1 * q1 * y3
    dydt[2] = k * (0.0083 - y4) - (Yos * q1 * y3)
    return dydt
@numba.njit
def sysY2(t, y, k=1, mu1=1, mu2=1, Ks1=1, Ks2=1, YXs1=1, YXs2=1, ms1=1, ms2=1, G=0, M=0):
    y1, y2, y3, y4 = y

    # Constant parameters
    MW_S1 = 180.06  # Molecular weight glucose [g/mol]
    MW_S2 = 150.13  # Molecular weight arabinose [g/mol]
    MW_B = 26.13  # Molecular weight of biomass in [g/C-mole] calculated from molecular composition

    OX = 0.615  # Oxygen content of biomass [mol/C-mol]
    HX = 1.832  # Hydrogen content of biomass [mol/C-mol]
    NX = 0.176  # Nitrogen content of biomass [mol/C-mol]

    # Reaction Stoichiometry
    # Pathway Glucose --> 1 glucose + a o2 + b NH3 --> b biomass + c CO2 + d H2O

    b = YXs1 * MW_S1 / MW_B  # Biomass and Nitrogen coefficient [mol/cmol]
    a = (3 * NX * b) / 4 - (HX * b) / 4 - b + (OX * b) / 2 + 6  # Oxygen coefficient [mol/cmol]
    c = 6 - b  # Carbon dioxide coefficient [mol/cmol]
    d = (3 * NX * b) / 2 - (HX * b) / 2 + 6  # H20 coefficient [mol/cmol]

    Y_O2_Glc = a / MW_S1 * 1000  # mmol/g Yield O2 glucose oxidative
    Y_CO2_Glc = c / MW_S1 * 1000  # mmol/g Yield CO2 glucose oxidative

    # Pathway Arabinose --> 1 arabinose + a o2 + b NH3 --> b biomass + c CO2 + d H2O

    b = YXs2 * MW_S2 / MW_B  # Biomass and Nitrogen coefficient [mol/cmol]
    a = (3 * NX * b) / 4 - (HX * b) / 4 - b + (OX * b) / 2 + 5  # Oxygen coefficient [mol/cmol]
    c = 5 - b  # Carbon dioxide coefficient [mol/cmol]
    d = (3 * NX * b) / 2 - (HX * b) / 2 + 6  # H20 coefficient [mol/cmol]

    Y_O2_Ara = a / MW_S2 * 1000  # mmol/g Yield O2 glucose oxidative
    Y_CO2_Ara = c / MW_S2 * 1000  # mmol/g Yield CO2 glucose oxidative
    q1 = mu1 * ((y1 / (y1 + Ks1)) - ms1 * YXs1)
    q2 = mu2 * ((y2 / (y2 + (Ks2 / Ks1 * y1) )) - ms2 * YXs2)

    dydt = np.empty(4)
    dydt[0] = -q1 * y3
    dydt[1] = -q2 * y3
    qS = q1 + q2

    # OUR calculation for state equation
    # Here, you would need to define x[0], q_Glc, and q_Ara elsewhere in your Python script
    OUR = (q1 * Y_O2_Glc + q2 * Y_O2_Ara)  # Oxygen uptake rate in mmol/h

    dydt[2] = (YXs1 * q1 + YXs2 * q2) * y3
    dydt[3] = k * (0.0083 - y4) - (OUR * y3)
    return dydt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import time as t
dfX = pd.read_excel('BioL data 3.xlsx', sheet_name='Biomass')
dfO= pd.read_excel('BioL data 3.xlsx', sheet_name='DO')
Gluc=pd.read_excel('BioL data 3.xlsx', sheet_name='Glucose').T
Gluc=Gluc[0].tolist()
Ara=pd.read_excel('BioL data 3.xlsx', sheet_name='Arabinose').T
Ara=Ara[0].tolist()
Time=pd.read_excel('BioL data 3.xlsx', sheet_name='Time')
Time=Time['t(h)'].tolist()
sumerror=0
#This loop is to iterate from X1 to X(last), so it DOES NOT need to star from 0 as always
#1,2,3,4,17,18,19,20,33,34,35,36
x=[17]
for i in x:
    # Define initial conditions

    sample='X'
    num=str(i)
    samplenum=sample+num
    DCW=dfX[samplenum]
    max_index = np.argmax(DCW)
    DCW[max_index:] = DCW[max_index]
    pO=dfO[samplenum]
    pO=pO/100
    G=Gluc[i]
    A=Ara[i]
    #A = 0.001
    y0 = [G, A, DCW[0], 0.0083]
    # Define time points to solve for
    t_span = (Time[0], 40)
    #t_span = (0,Time[-1])
    #time_points = Time

    args1 =[387.38778, 0.379258, 0.323276, 0.001034, 0.000914, 0.6, 0.578824, 0.003162, 0.049568, 4.9e-05]


    #sys2(t, y, mu1=1, mu2=1, Ks1=1, Ks2=1, YXs1=1, YXs2=1, ms1=1, ms2=1, G=0, M=0 ):
    sol = solve_ivp(sys2, t_span, y0,  method='LSODA',args = args)
    S1 = sol.y[0].tolist()
    S2 = sol.y[1].tolist()
    X = sol.y[2].tolist()
    O2 = sol.y[3].tolist()
    O2 = (np.array(O2) / 0.0083)
    time = sol.t.tolist()

    # Extract the solution for y and x
    y1 = sol.y[0]
    y2 = sol.y[1]
    y3 = sol.y[2]
    y4 = sol.y[3]


    # Plot the results

    #plt.plot(ProcTime,Glucose,label='S1 exp')

    #plt.plot(sol.t, y3, label='S3')
    #plt.plot(ProcTime,Mannose,label='S2 exp')
    #plt.plot(ProcTime,Xylose,label='S3 exp')
    plt.figure(figsize=(10, 8))  # Width=10 inches, Height=8 inches
    plt.plot(Time, DCW, label='X experimental', linestyle='--')
    plt.plot(sol.t, X, label='X')
    #plt.plot(sol.t, S1, label='Glucose')
    #plt.plot(sol.t, S2, label='Arabinose')
    plt.plot(Time, pO, label='pO2 experimental', linestyle='--')
    plt.plot(sol.t, O2, label='pO2')

    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.title('20 g/L Glucose and Arabinose (Non-Competitive inhibition)')  # Add title here
    plt.xlim(0, 40)
    plt.ylim(-1, 40)
    plt.legend()
    plt.show()