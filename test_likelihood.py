import numpy as np
from bayesian_hospital_in_a_box import likelihood
from scipy.stats import expon
from matplotlib import pyplot as plt

lambda_r = 4
lambda_l = 2.5
pc0 = 1
theta = np.array([lambda_r, lambda_l])

pr = expon(scale=1/lambda_r)
pl = expon(scale=1/lambda_l)

N = 1000
t_r_samples = pr.rvs(N)
t_l_samples = pl.rvs(N)

t_samples = t_r_samples + t_l_samples
t_bin_values, t_bins = plt.hist(t_samples, color='grey')[0:2]
t_range = np.linspace(np.min(t_bins), np.max(t_bins), 100)
pt = likelihood.p(theta, t_range, pc0)

plt.plot(t_range, pt / np.max(pt) * np.max(t_bin_values))


plt.show()
