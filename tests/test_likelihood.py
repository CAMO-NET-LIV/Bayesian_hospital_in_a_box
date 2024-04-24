import numpy as np
from bayesian_hospital_in_a_box import likelihood
from scipy.stats import expon, gamma
from matplotlib import pyplot as plt

def test_first_term(plots=False):
    """
    Tests the first term of the likelhood against histogram
    of results.
    """

    np.random.seed(42)

    # Problem parameters; noting that pc0=1 means we only
    # analyse the first term in the likelihood
    lambda_r = 1 / (2 * 60)
    lambda_l = 1 / (8 * 60)
    pc0 = 1
    theta = np.array([lambda_r, lambda_l])

    # Define generatrive distributions
    p_r = expon(scale=1/lambda_r)
    p_l = expon(scale=1/lambda_l)

    # Generate samples of total time
    N = int(1e5)
    t_r_samples = p_r.rvs(N)
    t_l_samples = p_l.rvs(N)
    t_samples = t_r_samples + t_l_samples

    # Evaluate histogram
    t_bin_values, t_bin_edges = np.histogram(t_samples)    
    t_bin_centres = (t_bin_edges[:-1] + t_bin_edges[1:]) / 2

    # Evaluate likelihood at bin centres of histogram
    p_t = likelihood.p(theta, t_bin_centres, pc0)
    
    # Standardise results for comparison
    t_bin_values = t_bin_values / np.sum(t_bin_values)
    p_t = p_t / np.sum(p_t)

    if plots:
        fig, ax = plt.subplots()
        ax.plot(t_bin_centres/60, t_bin_values, 'black', label='Histogram')
        ax.plot(t_bin_centres/60, p_t, 'red', label='Closed-form expression')
        ax.legend()
        ax.set_xlabel('Hours')
        plt.show()

    assert np.allclose(p_t, t_bin_values, atol=0.1)
    
def test_monte_carlo_estimate(plots=False):
    """
    Test Monte-Carlo estimate that appears in the likelihood
    against histogram of results.
    """
    np.random.seed(42)
   
    # Problem definition
    lambda_r = 4
    lambda_l = 2.5
    n = 5

    # Define generating distributions
    p_r = expon(scale=1/lambda_r)
    p_l = gamma(a=n, scale=1/lambda_l)

    # Generate samples and compute summatoin
    N = int(1e5)
    t_rl_samples = p_r.rvs(N) + p_l.rvs(N)

    # Evaluate histogram, standardising results
    t_rl_bin_values, t_rl_bin_edges = np.histogram(t_rl_samples)
    t_rl_bin_values = t_rl_bin_values / np.max(t_rl_bin_values)
    t_rl_bin_centres = (t_rl_bin_edges[:-1] + t_rl_bin_edges[1:]) / 2

    # Evaluate Monte-Carlo estiamte
    p_t_rl = np.zeros(len(t_rl_bin_centres))
    for i, t_rl in enumerate(t_rl_bin_centres):
        p_t_rl[i] = likelihood.exp_gamma_convolution_mc(lambda_r, lambda_l, n, t_rl)

    # Normalise results so we can compare them
    t_rl_bin_values = t_rl_bin_values / np.sum(t_rl_bin_values)
    p_t_rl = p_t_rl / np.sum(p_t_rl)    

    if plots:
        fig, ax = plt.subplots()
        ax.plot(t_rl_bin_centres, t_rl_bin_values, 'black', label='Histogram results')
        ax.plot(t_rl_bin_centres, p_t_rl, 'red', label='Monte Carlo estiamte')
        ax.legend()
        plt.show()

    assert np.allclose(t_rl_bin_values, p_t_rl, atol=0.1)

def test_p(plots=False):
    """
    Test overall expression for the likelihood (i.e. likelihood.p) against
    histogram results.
    """
    np.random.seed(42)

    # Example problem parameters; 
    lambda_r = 1 / (2 * 60)
    lambda_l = 1 / (8 * 60)
    pc0 = 0.2
    N_l_max = 3
    theta = np.array([lambda_r, lambda_l])

    # Define generatrive distributions
    p_r = expon(scale=1/lambda_r)
    p_l = expon(scale=1/lambda_l)

    # Initialise
    N = int(1e5)
    t_samples = np.zeros(N)

    for n in range(N):

        t_samples[n] = p_r.rvs()  # Initial transport tiem

        # Loop through the lab
        for i in range(N_l_max):
    
            # Add lab time
            t_samples[n] += p_l.rvs()
    
            # Can leave lab early
            u = np.random.rand()
            if u < pc0:
                break

    # Evaluate histogram
    t_bin_values, t_bin_edges = np.histogram(t_samples)    
    t_bin_centres = (t_bin_edges[:-1] + t_bin_edges[1:]) / 2

    # Evaluate likelihood at bin centres of histogram
    p_t = np.zeros(len(t_bin_centres))
    for i, t in enumerate(t_bin_centres):
        p_t[i] = likelihood.p(theta, t, pc0, N_l_max)

    # Normalise results so we can compare them
    t_bin_values = t_bin_values / np.sum(t_bin_values)
    p_t = p_t / np.sum(p_t)    

    if plots:
        fig, ax = plt.subplots()
        ax.plot(t_bin_centres/60, t_bin_values, 'black', label='Histogram results')
        ax.plot(t_bin_centres/60, p_t, 'red', label='Monte Carlo estimate')
        ax.set_xlabel('Hours')
        ax.legend()
        plt.show()

    assert np.allclose(p_t, t_bin_values, atol=0.1)
