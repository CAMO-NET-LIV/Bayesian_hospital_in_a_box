import numpy as np
from scipy.stats import expon
from scipy.stats import gamma

def p(theta, t, pc0):
    """
    Description
    -----------
        Evaluation of hospital-in-a-box likelihood
    
    Parameters
    ----------
        - theta : model parameters to be identified (lambda_r, lambda_l)
        - t : array of total times (i.e. observations)
        - pc0 : the probability that C = 0
    """
    
    # Extract model parameters
    lambda_r, lambda_l = theta[0], theta[1]
    
    # First term in the likelihood
    likelihood = lambda_r * lambda_l / (lambda_l - lambda_r)
    likelihood *= (np.exp(-lambda_r * t) - np.exp(-lambda_l * t))

    # If p(C=0) is less than 1 then the remaining terms need to be estimated
    # using Monte Carlo
    if pc0 < 1:
        pass

    likelihood *= pc0
    return likelihood