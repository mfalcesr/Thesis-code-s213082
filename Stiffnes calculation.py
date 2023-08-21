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
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def jacobian(func, y, t, args, eps=1e-5):
    """Calculate the Jacobian matrix using central differences."""
    n = len(y)
    jac = np.empty((n, n))

    for i in range(n):
        y_minus = y.copy()
        y_plus = y.copy()
        y_minus[i] -= eps
        y_plus[i] += eps
        jac[:, i] = (func(y_plus, t, args) - func(y_minus, t, args)) / (2 * eps)

    return jac


def sys2(y, t, args):
    y1, y2, y3, y4 = y
    k, mu1, mu2, Ks1, Ks2, YXs1, YXs2, ms1, ms2 = args

    q1 = mu1 * ((y1 / (y1 + Ks1)) - ms1 * YXs1)
    q2 = mu2 * ((y2 / (y2 + (Ks2 / Ks1 * y1))) - ms2 * YXs2)
    qS = q1 + q2
    Yos = ((-4 + YXs1 * 4.074) / -4) * (32 / 26.13)

    dydt = [
        -q1 * y3,
        -q2 * y3,
        (YXs1 * q1 + YXs2 * q2) * y3,
        k * (0.0083 - y4) - (Yos * qS * y3)
    ]
    return np.array(dydt)


args = [580.659993, 0.342284, 0.044018, 0.002771, 0.0001, 0.6, 0.5999, 0.00177, 0.04998]
y0 = [20, 20, 1.71, 0.0083]
t = np.linspace(0, 50, 1000)

sol = odeint(sys2, y0, t, args=(args,))

jacobians = [jacobian(sys2, sol[i], t[i], args) for i in range(len(t))]

# Compute stiffness as the ratio of the maximum to minimum non-zero eigenvalue
stiffnesses = []
for j in jacobians:
    eigvals = np.linalg.eigvals(j)
    max_eig = np.max(eigvals)
    min_nonzero_eig = np.min(np.abs(eigvals[eigvals != 0]))
    stiffnesses.append(max_eig / min_nonzero_eig)

# Plot the stiffness vs. time
plt.figure(figsize=(10, 6))
plt.plot(t, stiffnesses)
plt.xlim(0, 34)
plt.xlabel('Time')
plt.ylabel('Stiffness')
plt.title('Stiffness vs. Time')
plt.yscale('log')  # This sets the y-axis to a logarithmic scale
plt.grid(True, which="both", ls="--", c='0.65')  # Adjusts the grid to show on log scales
plt.show()