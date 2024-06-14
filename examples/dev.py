from bayesian_hospital_in_a_box.case_studies import case_study1
from bayesian_hospital_in_a_box.MCMC import MH, plot_MCMC_results
import numpy as np
from matplotlib import pyplot as plt

beta_l_true = 4
beta_r_true = 3
example1 = case_study1(beta_l_true, beta_r_true)
example1.generate_data(N=10)

# MCMC
mh = MH()

# Joint model
'''
theta0 = np.array([beta_l_true, beta_r_true])
theta0 = np.append(theta0, example1.tl_samples)
theta_joint = mh.sample(N=10000, theta0=theta0,
                        proposal_cov=0.01*np.eye(len(theta0)),
                        log_posterior=example1.joint_model_log_posterior)
plot_MCMC_results(ax=ax[0], samples=theta_joint[:, 0], color='red', label='Joint')
plot_MCMC_results(ax=ax[1], samples=theta_joint[:, 1], color='red', label='Joint')
'''

# Marginal model
theta_marginal = mh.sample(N=10000, theta0=np.array([beta_l_true, beta_r_true]),
                           proposal_cov=5*np.eye(2),
                           log_posterior=example1.marginal_log_posterior)

# Plot results from marginal
fig, ax = plt.subplots(ncols=2, nrows=2)
plot_MCMC_results(ax=ax[0][0], samples=theta_marginal[:, 0], color='black', label='Marginal')
plot_MCMC_results(ax=ax[0][1], samples=theta_marginal[:, 1], color='black', label='Marginal')
ax[1][0].plot(theta_marginal[:, 0], np.arange(0, len(theta_marginal)), color='black')
ax[1][1].plot(theta_marginal[:, 1], np.arange(0, len(theta_marginal)), color='black')
ax[0][0].plot(beta_l_true, 0, 'o', color='green')
ax[0][1].plot(beta_r_true, 0, 'o', color='green')
ax[0][1].legend()
plt.tight_layout()
fig, ax = plt.subplots()
ax.plot(theta_marginal[:, 0], theta_marginal[:, 1], '.', color='black', alpha=0.2)
ax.plot(beta_l_true, beta_r_true, 'o', color='green')
ax.grid()


plt.show()
