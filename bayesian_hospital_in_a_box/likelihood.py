import numpy as np
from scipy.stats import expon
from scipy.stats import gamma

def p(theta, t, pc0, N_l_max=None):
    """
    Description
    -----------
        Evaluation of hospital-in-a-box likelihood
    
    Parameters
    ----------
        - theta : model parameters to be identified (lambda_r, lambda_l)
        - t : array of total times (i.e. observations)
        - pc0 : the probability that C = 0
        - N_l_max : maximum possible number of cycles through the lab
    """
    
    # Extract model parameters
    lambda_r, lambda_l = theta[0], theta[1]
    
    # First term in the likelihood
    likelihood = lambda_r * lambda_l / (lambda_l - lambda_r)
    likelihood *= (np.exp(-lambda_r * t) - np.exp(-lambda_l * t))

    # If p(C=0) is less than 1 then the remaining terms need to be estimated
    # using Monte Carlo
    if pc0 < 1:

        # Probability that C = 1
        pc1 = 1 - pc0

        # Add additional terms that are appxoimated using Monte-Carlo
        for n in range(2, N_l_max + 1):
            likelihood += pc1**(n-1) * exp_gamma_convolution_mc(lambda_r, lambda_l, n, t)

    likelihood += pc1**(N_l_max) * exp_gamma_convolution_mc(lambda_r, lambda_l, N_l_max, t)

    return likelihood

def exp_gamma_convolution_mc(lambda_r, lambda_l, n, t_rl):
    """
    Description
    -----------
        Realises a Monte-Carlo estimate of the convolutation between the
        distributions p_r and p_l, where p_r = Exp(lambda_r) and
        p_l = Gamma(n, lambda_l) (see Appendix A of paper).
    
    Parameters
    ----------
        - lambda_r : rate parameter of p_r
        - lambda_l : rate parameter of p_l
        - n : shape parameter of p_l
        - t_rl : point where we approximate the convolution integral
    
    Returns
    -------
        - mean of F i.e. Monte Carlo estimate of convolution
    """
    
    # Define distributions
    p_r = expon(scale=1/lambda_r)
    p_l = gamma(a=n, scale=1/lambda_l)
    
    # Generate samples from exponential distribution
    N = 1000
    t_r_samples = p_r.rvs(N)
    
    # Realise Monte Carlo estimates of convolution integral at point t_rl
    F = _f(t_rl - t_r_samples, t_rl, p_l)
    return np.mean(F)

def _f(t_l, t_r_plus_l, p_l):
      """ Function that is equal to p_l(t_l) if of t_l are in [0, t_{r+l}]
        and 0 otherwise. Is used to realise Monte Carlo estimates in the function
        'exp_gamma_convolution_mc' and is described in Appendix A of the paper.
      """

      output = p_l.pdf(t_l)               # p_l evaluated over all t_l values
      zero_locations = t_l > t_r_plus_l   # Locations where t_l \notin [0, t_{r+l}]
      output[zero_locations] = 0          # Set appropriate locations equal to 0
      return output