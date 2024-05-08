import numpy as np
import pandas as pd
from bayesian_hospital_in_a_box.likelihood import logp
from matplotlib import pyplot as plt


# Load simulation data
df = pd.read_csv('results.csv')
t = df['time_completed'].values

# No. of simulation samples we will include
N_samples = 50

# Range of averge times over which we'll create contour plot
r_hrs = np.linspace(1, 10, 40)
l_hrs = np.linspace(1, 10, 40)

# Initialise contour
R_hrs = np.zeros([len(r_hrs), len(l_hrs)])
L_hrs = np.zeros([len(r_hrs), len(l_hrs)])
logP = np.ones([len(r_hrs), len(l_hrs)])

# Create contour with likelihood evaluations
n = 0
N_total = len(r_hrs) * len(l_hrs)
for i in range(len(r_hrs)):
    for j in range(len(l_hrs)):
        R_hrs[i, j] = r_hrs[i]
        L_hrs[i, j] = l_hrs[j]
        theta = [1 / (60 * r_hrs[i]),
                 1 / (60 * l_hrs[j])]
        logP[i, j] = logp(theta, t[:N_samples], pc0=0.2, N_l_max=3)

        n += 1
        print(n / N_total * 100, "percent done")

# Plot
plt.contourf(R_hrs, L_hrs, np.exp(logP))
plt.xlabel('Transport mean (hours)')
plt.ylabel('Lab mean (hours)')
plt.plot(np.array([2]), np.array([8]), 'o', color='red', label='True solution')
plt.legend()
plt.show()
