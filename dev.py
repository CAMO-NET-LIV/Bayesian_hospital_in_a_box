import numpy as np
from bayesian_hospital_in_a_box import likelihood
from scipy.stats import expon
from matplotlib import pyplot as plt
   
# Problem definition
lambda_r = 4
lambda_l = 2.5
pc0 = 0.7
N_l_max = 5
theta = np.array([lambda_r, lambda_l])

# Define generating distributions
p_r = expon(scale=1/lambda_r)
p_l = expon(scale=1/lambda_l)

# No. samples to generate for histrogram
N = 10000

t_samples = np.zeros(N)
for n in range(N):

    # Initialise with sample of transport time
    #t_samples[n] = p_r.rvs()

    # Loops through the lab
    for i in range(N_l_max):

        # Time through lab
        t_samples[n] += p_l.rvs()

        # Establish if sample need lab analysis
        u = np.random.rand()
        if u < pc0:
            break        

# Evaluate histogram, standardising results
t_bin_values, t_bin_edges = np.histogram(t_samples)
t_bin_values = t_bin_values / np.max(t_bin_values)
t_bin_centres = (t_bin_edges[:-1] + t_bin_edges[1:]) / 2

# Evaluate likelihood at bin centres of histogram
p_t = likelihood.p_total_lab(theta, t_bin_centres, pc0, N_l_max)
p_t = p_t / np.max(p_t)

fig, ax = plt.subplots()
ax.plot(t_bin_centres, t_bin_values, label='histogram')
ax.plot(t_bin_centres, p_t, label='likelihood estimate')
ax.legend()
plt.show()

#t = np.array([1, 2, 3, 4])
#pt = likelihood.p(theta, t, pc0, N_l_max)
#print(pt)