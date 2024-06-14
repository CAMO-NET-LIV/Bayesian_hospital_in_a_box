import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

class MH:
    def __init__(self):
        pass
    
    def sample(self, N, theta0, proposal_cov, log_posterior):

        # Initialise
        D = len(theta0)
        theta = np.zeros([N, D])
        theta[0, :] = theta0
        n_accept = 0

        # Proposal distribution
        p = multivariate_normal(cov=proposal_cov)

        for i in range(N-1):
            
            # Proposal
            theta_dash = theta[i, :] + p.rvs()
            
            # Accept / reject
            loga = log_posterior(theta_dash) - log_posterior(theta[i, :])
            u = np.random.rand()
            if np.log(u) < loga:
                theta[i+1, :] = theta_dash
                n_accept += 1
            else:
                theta[i+1, :] = theta[i, :]

        print("Acceptance rate = " + str(n_accept / N * 100) + " percent")
        return theta

def plot_MCMC_results(ax, samples, range=None, color='black', label=''):

    if range is None:
        range = np.linspace(np.min(samples), np.max(samples), 50)

    hist_values, bin_edges = np.histogram(samples, bins=range)
    hist_values = hist_values / np.sum(hist_values)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, hist_values, color=color, label=label)
