from bayesian_hospital_in_a_box.case_studies import case_study1
from bayesian_hospital_in_a_box.MCMC import MH, plot_MCMC_results
import numpy as np
from matplotlib import pyplot as plt
import pymc as pm

# Initialise and generate data
N = 10
Ns = 5000
beta_l_true = 10
beta_r_true = 1
example1 = case_study1(beta_l_true, beta_r_true)
#example1.generate_data(N)
example1.generate_des_data(N)

# MCMC
trace_marginal_posterior = example1.sample_marginal_posterior(Ns)  # Sample from marginal
trace_joint_posterior = example1.sample_joint_posterior(Ns)        # Sample from joint

# Plot samples of beta_l
r = np.linspace(0, 20, 100)
fig, ax = plt.subplots()
plot_MCMC_results(ax, trace_marginal_posterior['beta_l'], r, color='black', label='Marginal Posterior (PyMC)')
plot_MCMC_results(ax, trace_joint_posterior['beta_l'], r, color='green', label='Joint Posterior (PyMC)')
ax.plot(beta_l_true, 0, 'o', color='green')
ax.set_xlabel('beta_l')
ax.legend()
ax.grid()

# Plot samples of beta_r
r = np.linspace(0, 20, 100)
fig, ax = plt.subplots()
plot_MCMC_results(ax, trace_marginal_posterior['beta_r'], r, color='black', label='Marginal Posterior (PyMC)')
plot_MCMC_results(ax, trace_joint_posterior['beta_r'], r, color='green', label='Joint Posterior (PyMC)')
ax.plot(beta_r_true, 0, 'o', color='green')
ax.set_xlabel('beta_r')
ax.legend()
ax.grid()