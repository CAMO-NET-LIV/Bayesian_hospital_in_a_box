import numpy as np
from scipy.stats import expon
import salabim as sim


class SpecimenCaseStudy1(sim.Component):
    def __init(self, beta_l_true, beta_r_true, log, **kwargs):
        super().__init__(**kwargs)
        self.beta_l_true = beta_l_true
        self.beta_r_true = beta_r_true
        self.log = log
        self.log[self.name()] = {}

    def process(self):
        collection_t = self.env.now()
        self.log[self.name()]['time_collected'] = collection_t
        yield self.hold(sim.Exponential(rate=1/self.beta_r_true).sample())
        self.log[self.name()]['time_received'] = self.env.now() - collection_t
        yield self.hold(sim.Exponential(rate=1/self.beta_l_true).sample())
        self.log[self.name()]['time_lab'] = self.env.now() - collection_t


class case_study1():

    def __init__(self, beta_l_true, beta_r_true):
        self.beta_l_true = beta_l_true
        self.beta_r_true = beta_r_true
        self.beta_l_prior = expon(scale=10)
        self.beta_r_prior = expon(scale=10)
    
    def generate_data(self, N):
        pr = expon(scale=self.beta_r_true)  # Distribution over transport time
        pl = expon(scale=self.beta_l_true)  # Distribution over lab time
        self.tr_samples = pr.rvs(N)         # Samples of transport time
        self.tl_samples = pl.rvs(N)         # Samples of lab time
        self.t_samples = self.tr_samples + self.tl_samples  # Samples of total time
        self.N = N

    def generate_des_data(self, N):
        log = {}
        env = sim.Environment(trace=False)
        sim.ComponentGenerator(SpecimenCaseStudy1,
                               beta_l_true=self.beta_l_true,
                               beta_r_true=self.beta_r_true,
                               iat=10,
                               number=N,
                               log=log)
        env.run()
        self.tr_samples = np.array([log[specimen.name()]['time_received'] for specimen in log])
        self.tl_samples = np.array([log[specimen.name()]['time_lab'] - log[specimen.name()]['time_received']
                                    for specimen in log])
        self.t_samples = np.array([log[specimen.name()]['time_lab'] for specimen in log])
        self.N = N

    def __joint_model_log_prior(self, beta_l, beta_r, t_l):
        """
        Log prior for the joint model
        """

        # Log-prior over beta_l and beta_r
        ans = self.beta_l_prior.logpdf(beta_l)
        ans += self.beta_r_prior.logpdf(beta_r)
        
        # Log-prior over lab times
        pt = expon(scale=beta_l)  
        ans += np.sum(pt.logpdf(t_l))

        return ans
    
    def __joint_model_log_likelihood(self, beta_l, beta_r, t_l):
        """
        Log-likelihood for the joint model
        """

        # Log-likelihood
        p = expon(scale=beta_r)
        ans = np.sum(p.logpdf(self.t_samples - t_l))

        return ans
    
    def joint_model_log_posterior(self, theta):

        # Extract parameters
        beta_l, beta_r, t_l = theta[0], theta[1], theta[2:]
        
        # Log posterior
        ans = self.__joint_model_log_prior(beta_l, beta_r, t_l)
        ans += self.__joint_model_log_likelihood(beta_l, beta_r, t_l)

        return ans
    
    def __marginal_model_log_prior(self, beta_l, beta_r):
        """
        Log prior for the marginal model
        """
        ans = self.beta_l_prior.logpdf(beta_l)
        ans += self.beta_r_prior.logpdf(beta_r)
        return ans

    def __marginal_model_log_likelihood(self, beta_l, beta_r):
        """
        Log-likelhood for the marginal model (here we compute
        it before taking the logarithm at the end)
        """

        ans = ((beta_r * beta_l)**-1 * (beta_r**-1 - beta_l**-1)**-1)**self.N
        for i in range(self.N):
            ans *= (np.exp(-beta_l**-1 * self.t_samples[i]) -
                    np.exp(-beta_r**-1 * self.t_samples[i]))

        return np.log(ans)

    def marginal_log_posterior(self, theta):
        beta_l, beta_r = theta[0], theta[1]
        ans = self.__marginal_model_log_prior(beta_l, beta_r)
        ans += self.__marginal_model_log_likelihood(beta_l, beta_r)
        return ans
