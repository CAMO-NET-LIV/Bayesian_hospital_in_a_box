import numpy as np
import pandas as pd
from bayesian_hospital_in_a_box.likelihood import logp
from matplotlib import pyplot as plt

# Load simulation data
df = pd.read_csv('results.csv')
t = df['time_completed'].values

# No. of simulation samples we will include
t = t[:50]

# Set burn-in
burn_in = 100

# Initial theta
r_hrs = 3
l_hrs = 3
theta0 = np.array([1 / (60 * r_hrs), 1 / (60 * l_hrs)])

# MCMC
N = 1000
theta = np.zeros([N, 2])
theta[0] = theta0
logp_now = logp(theta0, t, pc0=0.2, N_l_max=3)
for i in range(N-1):

    # Propose sample
    theta_dash = theta[i, :] + 1e-3 * np.random.randn(2)
    if np.sum(theta_dash < 0) > 0:
        theta[i+1, :] = theta[i, :]
    elif np.sum(theta_dash > 10) > 0:
        theta[i+1, :] = theta[i, :]
    else:
        logp_dash = logp(theta_dash, t, pc0=0.2, N_l_max=3)

        # Accept / Reject
        loga = logp_dash - logp_now
        u = np.random.rand()
        if u < np.exp(loga):
            theta[i+1, :] = theta_dash
            logp_now = np.copy(logp_dash)
        else:
            theta[i+1, :] = theta[i, :]

# Transform resutls
theta_hrs = np.zeros([N, 2])
theta_hrs[:, 0] = 1 / (60 * theta[:, 0])
theta_hrs[:, 1] = 1 / (60 * theta[:, 1])

# Plot results
fig, ax = plt.subplots()
ax.plot(theta_hrs[:burn_in, 0], theta_hrs[:burn_in, 1], color='red', alpha=0.5)
ax.plot(theta_hrs[burn_in:, 0], theta_hrs[burn_in:, 1], color='black', alpha=0.5)
ax.plot(theta_hrs[burn_in:, 0], theta_hrs[burn_in:, 1], 'o', color='black', alpha=0.5)
ax.grid()
ax.set_xlim([1, 10])
ax.set_ylim([1, 10])
ax.set_xlabel('Transport mean (hours)')
ax.set_ylabel('Lab mean (hours)')

fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].hist(theta_hrs[burn_in:, 0])
ax[0][1].hist(theta_hrs[burn_in:, 1])
ax[1][0].plot(theta_hrs[:, 0], np.arange(0, N), 'black')
ax[1][0].plot(theta_hrs[:burn_in, 0], np.arange(0, burn_in), 'red')
ax[1][1].plot(theta_hrs[:, 1], np.arange(0, N), 'black')
ax[1][1].plot(theta_hrs[:burn_in, 1], np.arange(0, burn_in), 'red')

plt.show()