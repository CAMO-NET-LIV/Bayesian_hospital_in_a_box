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
        
        # Define distribution over transport time
        p_r = expon(scale=1/lambda_r)

        # Probability that C = 1
        pc1 = 1 - pc0

        # No. Monte Carlo samples
        N_MC = 10000

        for n in range(2, N_l_max + 1):
            
            p_l = gamma(a=n, scale=1/lambda_l)
            
            # Normalising const.
            norm_const = 0
            for k in range(N_l_max):
                norm_const += pc1**(n-1) * pc0

            # Generate samples from p_r
            t_r_samples = p_r.rvs(N_MC)
            
            # Compute terms for all values of t
            c = pc1**(n-1)
            for i in range(len(t)):
                likelihood[i] += pc1 * np.mean(_f(t[i] - t_r_samples, t[i], p_l))
            

    likelihood *= pc0
    return likelihood

def p_total_lab(theta, t, pc0, N_l_max=None):
    """
    Description
    -----------
    temp - for debugging
    """
    
    # Extract model parameters
    lambda_r, lambda_l = theta[0], theta[1]
    
    # Probability that C = 1
    pc1 = 1 - pc0

    # No. Monte Carlo samples
    N_MC = 10000

    likelihood = np.zeros(len(t))
    for n in range(1, N_l_max + 1):
            
        p_l = gamma(a=n, scale=1/lambda_l)
            
        likelihood += pc0 * pc1**(n-1) * p_l.pdf(t)
        
    return likelihood

def _f(t_l, t_r_plus_l, p_l):
      """ Function that is equal to p_l(_tl) if t_l in [0, t_{r+l}] and
        0 otherwise.
      """

      output = p_l.pdf(t_l)               # p_l evaluated over all t_l values
      zero_locations = t_l > t_r_plus_l   # Locations where t_l \notin [0, t_{r+l}]
      output[zero_locations] = 0          # Set appropriate locations equal to 0
      return output