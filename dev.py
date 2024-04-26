from bayesian_hospital_in_a_box import likelihood
import numpy as np
from scipy.stats import expon
from matplotlib import pyplot as plt

# Example problem parameters; 
lambda_l = 1 / (8 * 60)
pc0 = 0.2
N_l_max = 3

# Define generatrive distributions
p_l = expon(scale=1/lambda_l)

# Initialise
N = int(1e5)
t_l_samples = np.zeros(N)

for n in range(N):

    t_l_samples[n] = 0

    # Loop through the lab
    for i in range(N_l_max):
    
        # Add lab time
        t_l_samples[n] += p_l.rvs()
    
        # Can leave lab early
        u = np.random.rand()
        if u < pc0:
            break

# Evaluate histogram
t_l_bin_values, t_l_bin_edges = np.histogram(t_l_samples, bins=20)
t_l_bin_centres = (t_l_bin_edges[:-1] + t_l_bin_edges[1:]) / 2

# Evaluate likelihood at bin centres of histogram
p_t_l = np.zeros(len(t_l_bin_centres))
for i, t_l in enumerate(t_l_bin_centres):
    p_t_l[i] = likelihood.p_total_lab_time(t_l, lambda_l, pc0, N_l_max=3)

# Normalise results so we can compare them
t_l_bin_values = t_l_bin_values / np.sum(t_l_bin_values)
p_t_l = p_t_l / np.sum(p_t_l)    

fig, ax = plt.subplots()
ax.plot(t_l_bin_centres/60, t_l_bin_values, 'black', label='Histogram results')
ax.plot(t_l_bin_centres/60, p_t_l, 'red', label='Monte Carlo estimate')
ax.set_xlabel('Hours')
ax.legend()
ax.set_title('Total lab time')
plt.show()
