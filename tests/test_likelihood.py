import numpy as np
from bayesian_hospital_in_a_box import likelihood
from scipy.stats import expon
from matplotlib import pyplot as plt

def test_first_term(plots=False):
    """
    Tests the first term of the likelhood against histogram
    of results.
    """

    np.random.seed(42)

    # Problem parameters; noting that pc0=1 means we only
    # analyse the first term in the likelihood
    lambda_r = 4
    lambda_l = 2.5
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

    # Evaluate histogram, standardising results
    t_bin_values, t_bin_edges = np.histogram(t_samples)
    t_bin_values = t_bin_values / np.max(t_bin_values)
    t_bin_centres = (t_bin_edges[:-1] + t_bin_edges[1:]) / 2

    # Evaluate likelihood at bin centres of histogram
    p_t = likelihood.p(theta, t_bin_centres, pc0)
    p_t = p_t / np.max(p_t)

    if plots:
        fig, ax = plt.subplots()
        ax.plot(t_bin_centres, t_bin_values)
        ax.plot(t_bin_centres, p_t)
        plt.show()

    assert np.allclose(p_t, t_bin_values, atol=0.15)