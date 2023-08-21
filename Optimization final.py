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
import numba

#ADD CONDITION THAT IF SUBSTRATE IS GOING BELOW 0 LIMIT THE OVERSTEP
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


def interpolarraysugar(sugarexp, timeexp, sugarsol, timesol):
    end = sugarexp.index(0)
    my_list = range(len(sugarexp))
    for i in my_list[1:end]:
        target_number = sugarexp[i]
        closest_below = max((x for x in sugarsol if x <= target_number), default=None)
        closest_above = min((x for x in sugarsol if x >= target_number), default=None)
        below_index = sugarsol.index(closest_below)
        above_index = sugarsol.index(closest_above)
        x = [sugarsol[below_index], sugarsol[above_index]]
        y = [timesol[below_index], timesol[above_index]]
        x_new = target_number
        y_new = np.interp(x_new, x, y)

    return y_new, x_new
def interpolarraytime(sugarexp, timeexp, sugarsol, timesol):
    x_new = []
    y_new = []
    my_list = range(len(sugarexp))
    print(len(sugarexp))
    print(len(timeexp))
    print(len(sugarsol))
    print(len(timesol))

    for i in range(len(sugarexp)):
        target_number = timeexp[i]
        try:
            closest_below = max((x for x in timesol if x <= target_number), default=None)
            closest_above = min((x for x in timesol if x >= target_number), default=None)
            below_index = timesol.index(closest_below)
            above_index = timesol.index(closest_above)
            x = [timesol[below_index], timesol[above_index]]
            y = [sugarsol[below_index], sugarsol[above_index]]
            x_new.append(target_number)
            y_new.append(np.interp(target_number, x, y))
        except ValueError:
            continue

    return x_new, y_new
def newsol(sugarexp,timeexp,sugarsol,timesol):
    newsugarsol=[]
    newtimesol=[]
    newsugarsol.append(sugarexp[0])
    newtimesol.append(timesol[0])
    #check that this interpolation is correct
    x_new,y_new=interpolarraytime(sugarexp, timeexp, sugarsol, timesol)
    for i in range(len(y_new)):
        newsugarsol.append(y_new[i])
        newtimesol.append(x_new[i])
    return newsugarsol,newtimesol
def errorcalc (sugarexp,timeexp,sugarsol,timesol):
    newsugarsol,newtimesol=newsol(sugarexp,timeexp,sugarsol,timesol)
    errorlist=[]
    sumerror=0
    for i in range(len(newsugarsol)):
        error=(newsugarsol[i]-sugarexp[i])
        sumerror=error**2+sumerror
    return sumerror
def callback(xk, info):
    print(f"Objective function value: {info['fun']}")
    print(f"Iteration number: {info['nit']}")
def weightederror(array, time_series, threshold, W1, W2):
    result = np.copy(array)
    mask = time_series <= threshold
    result[mask] *= W1
    result[~mask] *= W2
    return result
def weightederrorsignal(array, biomass_series, time_series, threshold, W1, W2):
    result = np.copy(array)

    # Find the position of the value in biomass_series closest to the threshold
    position = np.argmin(np.abs(np.array(biomass_series) - threshold))

    # Convert time_series to a numpy array
    time_series = np.array(time_series)

    # Use the corresponding value from time_series as the threshold for the mask
    mask_threshold = time_series[position]
    mask = time_series <= mask_threshold

    result[mask] *= W1
    result[~mask] *= W2

    return result
def interpolate_errors(errorlist):
    interpolated_errors = [errorlist[0]]  # Start with the first value

    for i in range(1, len(errorlist)):
        if errorlist[i] > errorlist[0]:
            # Interpolate the value using the previous and next points
            prev_error = errorlist[i-1]
            next_error = errorlist[i+1] if i+1 < len(errorlist) else None

            if next_error is not None:
                interpolated_value = (prev_error + next_error) / 2.0
            else:
                interpolated_value = prev_error

            interpolated_errors.append(interpolated_value)
        else:
            interpolated_errors.append(errorlist[i])

    return interpolated_errors

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
dfX = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Biomass')
dfO = pd.read_excel('BioL data Gluc.xlsx', sheet_name='DO')
Gluc = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Glucose').T
Gluc = Gluc[0].tolist()
Ara = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Arabinose').T
Ara = Ara[0].tolist()
Time = pd.read_excel('BioL data Gluc.xlsx', sheet_name='Time')
Time = Time['t(h)'].tolist()
# 300, 0.14, 0.074, 0.049, 0.0022, 0.094, 0.046, 0.576, 0.596, 0.545, 0.0043, 0.011, 0.0038
realpar = [300, 0.14, 0.074, 0.049, 0.0022, 0.094, 0.046, 0.576, 0.596, 0.545, 0.0043, 0.011, 0.0038]
errorlist=[]
# Define the objective function optimization
def objective(x):
    k, mu1, mu2, Ks1, Ks2, YXs1, YXs2, ms1, ms2 , X= x
    Time=pd.read_excel('BioL data Gluc.xlsx', sheet_name='Time')
    Time=Time['t(h)'].tolist()
    sumerror = 0
    #dfX.shape[1]
    #put here all the same wells corresponding to the same strain and conditions
    x=[33]
    for i in x:
        sample = 'X'
        num = str(i)
        samplenum = sample + num
        DCW = dfX[samplenum]
        max_index = np.argmax(DCW)
        DCW[max_index:] = DCW[max_index]
        pO = dfO[samplenum]
        pO = pO / 100
        G = Gluc[i]
        M = Ara[i]
        #y0 = [Gluc[i], Ara[i], 1.34, 0.0083]
        y0 = [Gluc[i], Ara[i], DCW[0], 0.0083]
        args = (0.14, 0.074, 0.0022, 0.094, 0.576, 0.596, 0.0043, 0.011, G, M)
        # Solve the system of differential equations
        sol = solve_ivp(sys2, t_span, y0, method='LSODA', dense_output=True, args=(k, mu1, mu2, Ks1, Ks2, YXs1, YXs2, ms1, ms2,X))
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
        e1=(DCW-X)**2
        e2=(pO-O2)**2
        e1 = weightederrorsignal(e1, DCW, Time, (2*DCW[0]), 0.01, 1)
        #e2= weightederror(e2,Time,17,0.01,1)
        error=e1+e2
        sumerror=sum(error)
        print(sumerror)
        if i == x[-1]:
          errorlist.append(sumerror)
    return sumerror


# Define initial conditions
t_span = (0, 50)
#sys2(t, y,k ,mu1=1, mu2=1, Ks1=1, Ks2=1, YXs1=1, YXs2=1, ms1=1, ms2=1, G=0, M=0 ):
# Define bounds for the parameters to be optimized
bounds = [(25,600), (0.001, 0.4), (0.001, 0.4), (0.0001, 0.35), (0.0001, 0.35), (0.3, 0.6), (0.3, 0.6), (0.0001, 0.05), (0.0001, 0.05),(-10, 10)]
def callback(xk):
    #print("Current x:", xk)
    print("Number of function evaluations:", callback.nfev)
    callback.nfev += 1

callback.nfev = 0
max_iterations = 2000

# Define the options with maxiter set to max_iterations
options = {'maxiter': max_iterations}
# Perform optimization
result = minimize(objective, [497.3589703, 0.420367052, 0.084062344, 0.2, 0.162636562, 0.599999898, 0.304049926, 0.006667857, 0.001069403, 4.9e-05],
                  bounds=bounds,method='Nelder-Mead',callback=callback,options=options)
# Print the optimized parameters
print(result.x.tolist())# Print the optimized parameters
print(result)
x=[]
for i in result.x.tolist():
    x.append(round(i, 6))
print(x)