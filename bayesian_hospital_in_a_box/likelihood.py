import numpy as np
from scipy.stats import expon
from scipy.stats import gamma

def logp(theta, t, pc0, N_l_max=None):
    """
    Description
    -----------
        Evaluation of hospital-in-a-box log-likelihood
    
    Parameters
    ----------
        - theta : model parameters to be identified (lambda_r, lambda_l)
        - t : array of total times (i.e. observations). Note this is required
            to be an array of samples from the same hospital simulation
        - pc0 : the probability that C = 0
        - N_l_max : maximum possible number of cycles through the lab
    """
 
    # Extract model parameters
    lambda_r, lambda_l = theta[0], theta[1]
    
    # First term in the likelihood
    likelihood = _pn(n=1, pc0=pc0, N_l_max=N_l_max)
    if lambda_r == lambda_l:
        likelihood *= lambda_r**2 * t * np.exp(-lambda_r * t)
    else:
        likelihood *= lambda_r * lambda_l / (lambda_l - lambda_r)
        likelihood *= (np.exp(-lambda_r * t) - np.exp(-lambda_l * t))

    # If p(C=0) is less than 1 then the remaining terms need to be estimated
    # using Monte Carlo
    if pc0 < 1:

        # Add additional terms that are appxoimated using Monte-Carlo
        for n in range(2, N_l_max + 1):
            likelihood += (_pn(n=n, pc0=pc0, N_l_max=N_l_max) *
                           exp_gamma_convolution_mc(lambda_r, lambda_l, n, t))

    return np.sum(np.log(likelihood))

def p_total_lab_time(t_l, lambda_l, pc0, N_l_max):
    """
    Description
    -----------
        Expression for the probability of total lab time

    Parameters
    ----------
        t_l : total lab time
        lambda_l : rate parameter of p_l
        pc0 : the probability that C = 0
        N_l_max : maximum possible number of cycles through the lab
    
    Returns
    -------
        p_t_l : probability of total lab time equal to t_l
    """

    p_t_l = 0
    for n in range(1, N_l_max + 1):
        p_l = gamma(a=n, scale=1/lambda_l)
        p_t_l += p_l.pdf(t_l) * _pn(n, pc0, N_l_max)
    return p_t_l

def _pn(n, pc0, N_l_max):
    """ Function that evaluates the probability of n lab-visits
    """

    pc1 = 1 - pc0
    pn = pc1**(n - 1) * pc0
    if n == N_l_max:
        pn += pc1**N_l_max
    return pn

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
 
    # Dealing with input being float64 rather than array
    if isinstance(t_rl, np.float64):
        t_rl = np.array([t_rl])

    # Generate samples from exponential distribution
    N_MC = 1000
    N = len(t_rl)
    t_r_samples = p_r.rvs(N_MC)
 
    F = np.zeros([N_MC, N])
    for n in range(N):
        F[:, n] = t_rl[n] - t_r_samples

    # Realise Monte Carlo estimates of convolution integral at point t_rl
    F = p_l.pdf(F)
    return np.mean(F, axis=0)
