import numpy as np
import pandas as pd
from bayesian_hospital_in_a_box.likelihood import logp
from matplotlib import pyplot as plt

"""
Example where we run Metropolis Hastings to estimate
hospital parameters.
"""

# Load simulation data
df = pd.read_csv('results.csv')
t = df['time_completed'].values

# No. of simulation samples we will include
t = t[:50]

# Set burn-in
burn_in = 50

# Limits of uniform prior (which correspond to average times
# of between 1 and 10 hours)
prior_lower = 1 / (60 * 10)
prior_upper = 1 / 60

# Initial theta
theta0 = np.array([0.002, 0.002])

# MCMC
N = 200
theta = np.zeros([N, 2])
theta[0] = theta0
logp_now = logp(theta0, t, pc0=0.2, N_l_max=3)
for i in range(N-1):

    # Propose sample
    theta_dash = theta[i, :] + 1e-3 * np.random.randn(2)
    
    # Prior limits
    if np.sum(theta_dash < 0) > 0 or np.sum(theta_dash > prior_upper) > 0:
        theta[i+1, :] = theta[i, :]
        
    # Accept / Reject step
    else:
        logp_dash = logp(theta_dash, t, pc0=0.2, N_l_max=3)
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
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0][0].hist(theta_hrs[burn_in:, 0])
ax[0][1].hist(theta_hrs[burn_in:, 1])
ax[1][0].plot(theta_hrs[:, 0], np.arange(0, N), 'black')
ax[1][0].plot(theta_hrs[:burn_in, 0], np.arange(0, burn_in), 'red')
ax[1][0].plot(np.repeat(2, N), np.arange(0, N), 'green')
ax[1][1].plot(theta_hrs[:, 1], np.arange(0, N), 'black')
ax[1][1].plot(theta_hrs[:burn_in, 1], np.arange(0, burn_in), 'red', label='Burn in')
ax[1][1].plot(np.repeat(8, N), np.arange(0, N), 'green', label='True values')
ax[1][0].set_xlabel('Transport mean (hours)')
ax[1][1].set_xlabel('Lab mean (hours)')
plt.tight_layout()
plt.show()
