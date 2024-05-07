import numpy as np
import pandas as pd
from bayesian_hospital_in_a_box.likelihood import p
from matplotlib import pyplot as plt


# Load simulation data
df = pd.read_csv('results.csv')
t = df['time_completed'].values

# No. of simulation samples we will include
N_samples = 5

# Range of averge times over which we'll create contour plot
r_hrs = np.linspace(1, 10, 20)
l_hrs = np.linspace(1, 10, 20)

# Initialise contour
R_hrs = np.zeros([len(r_hrs), len(l_hrs)])
L_hrs = np.zeros([len(r_hrs), len(l_hrs)])
P = np.ones([len(r_hrs), len(l_hrs)])

# Create contour with likelihood evaluations
for i in range(len(r_hrs)):
    for j in range(len(l_hrs)):
        R_hrs[i, j] = r_hrs[i]
        L_hrs[i, j] = l_hrs[j]
        theta = [1 / (60 * r_hrs[i]),
                 1 / (60 * l_hrs[j])]
        #for n in range(N_samples):
        #    P[i, j] *= p(theta, t[n], pc0=0.2, N_l_max=3)
        P[i, j] = p(theta, t[:N_samples], pc0=0.2, N_l_max=3)

# Plot
plt.contourf(R_hrs, L_hrs, P)
plt.xlabel('Transport mean (hours)')
plt.ylabel('Lab mean (hours)')
plt.show()
