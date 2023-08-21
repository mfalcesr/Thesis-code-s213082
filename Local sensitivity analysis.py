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
from scipy.integrate import solve_ivp
def sys2(t, y,k=1, mu1=1, mu2=1, Ks1=1, Ks2=1, YXs1=1, YXs2=1, ms1=1, ms2=1, G=0, M=0 ):
    y1, y2, y3, y4 = y

    q1 = mu1 * ((y1 / (y1 + Ks1)) - ms1 * YXs1)
    q2 = mu2 * ((y2 / (y2 + (Ks2 / Ks1 * y1))) - ms2 * YXs2)
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


# Time array for the simulation

y0 = [20, 20, 1.71, 0.0083]

t_span = (0, 50)

# Create an array of time points at which the solution should be stored
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# Original arguments k, mu1, mu2, Ks1, Ks2, YXs1, YXs2, ms1, ms2,
args = [580.659993, 0.342284, 0.044018, 0.002771, 0.0001, 0.6, 0.5999, 0.00177, 0.04998]

# Store all solutions
solutions = {}

# Solve with original args
sol = solve_ivp(sys2, t_span, y0, method='LSODA', args=args, t_eval=t_eval)
S1 = sol.y[0].tolist()
S2 = sol.y[1].tolist()
X = sol.y[2].tolist()
O2 = (np.array(sol.y[3].tolist()) / 0.0083).tolist()
time = sol.t.tolist()

# Store original solution
solutions['original'] = {
    'S1': S1,
    'S2': S2,
    'X': X,
    'O2': O2,
    'time': time
}

# Loop over each argument
for i in range(len(args)):
    # Make a copy of the original args
    temp_args = args.copy()

    # Increase the i-th argument by 1%
    temp_args[i] *= 1.01

    # Solve with modified args
    sol = solve_ivp(sys2, t_span, y0, method='LSODA', args=temp_args,dense_output=True)


   # Get the solution at the same time steps as in the original solution
    t_eval = solutions['original']['time']
    y = sol.sol(t_eval)

    # Store the solution
    S1 = y[0].tolist()
    S2 = y[1].tolist()
    X = y[2].tolist()
    O2 = (np.array(y[3].tolist()) / 0.0083).tolist()
    time = t_eval  # The time vector should be the same as in the original solution

    # Store the solution
    solutions[f'args_{i}'] = {
        'S1': S1,
        'S2': S2,
        'X': X,
        'O2': O2,
        'time': time
    }
keys = list(solutions.keys())
print(keys)
veS1=[]
veS2=[]
veX=[]
veO2=[]
for i in range(len(keys)):
 diffS1 = np.array(solutions[keys[i]]['S1']) - np.array(solutions['original']['S1'])
 diffS2 = np.array(solutions[keys[i]]['S2']) - np.array(solutions['original']['S2'])
 diffX = np.array(solutions[keys[i]]['X']) - np.array(solutions['original']['X'])
 diffO2 = np.array(solutions[keys[i]]['O2']) - np.array(solutions['original']['O2'])
 veS1.append(diffS1)
 veS2.append(diffS2)
 veX.append(diffX)
 veO2.append(diffO2)

import matplotlib.pyplot as plt

# Assuming time is the same for all solutions and 'original'
time = np.array(solutions['original']['time'])
labels = ['base', 'KLa', 'mu1', 'mu2', 'Ks1', 'Ks2', 'YXs1', 'YXs2', 'ms1', 'ms2', 's1']

# Create a new figure
plt.figure(figsize=(10, 8))  # Width=10 inches, Height=8 inches
for i in range(len(veS1)):
    plt.plot(time, veS1[i], label=labels[i])

# Add a legend
plt.legend()

# Add x and y labels
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Evolution of differences respect to baseline respect to S1')
plt.xlim(0, 30)

# Show the plot
plt.show()

# Assuming time is the same for all solutions and 'original'
time = np.array(solutions['original']['time'])
labels = ['base', 'KLa', 'mu1', 'mu2', 'Ks1', 'Ks2', 'YXs1', 'YXs2', 'ms1', 'ms2', 's1']

# Create a new figure
plt.figure(figsize=(10, 8))  # Width=10 inches, Height=8 inches
for i in range(len(veS2)):
    plt.plot(time, veS2[i], label=labels[i])

# Add a legend
plt.legend()

# Add x and y labels
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Evolution of differences respect to baseline respect to baseline respect to S2')
plt.xlim(0, 30)

# Show the plot
plt.show()

# Assuming time is the same for all solutions and 'original'
time = np.array(solutions['original']['time'])
labels = ['base', 'KLa', 'mu1', 'mu2', 'Ks1', 'Ks2', 'YXs1', 'YXs2', 'ms1', 'ms2', 's1']

# Create a new figure
plt.figure(figsize=(10, 8))  # Width=10 inches, Height=8 inches
for i in range(len(veS1)):
    plt.plot(time, veX[i], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Evolution of differences respect to baseline respect to baseline respect to X')
# Add a legend
plt.legend()

# Add x and y labels
plt.xlim(0, 30)

# Show the plot
plt.show()

# Create a new figure
plt.figure(figsize=(10, 8))  # Width=10 inches, Height=8 inches
for i in range(len(veS1)):
    plt.plot(time, veO2[i], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Evolution of differences respect to baseline respect to baseline respect to O2')
# Add a legend
plt.legend()

# Add x and y labels
plt.xlim(9.2, 9.6)

# Show the plot
plt.show()