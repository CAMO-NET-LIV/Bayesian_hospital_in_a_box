import numpy as np
from scipy.stats import expon
import salabim as sim
import pymc as pm

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
        self.beta_l_prior_scale = 10
        self.beta_r_prior_scale = 10
    
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

    def __marginal_model_log_likelihood(self, value, beta_l, beta_r):
        J = 1
        lambda_l = 1 / beta_l
        lambda_r = 1 / beta_r
        for i in range(N):
            J = J * lambda_r * lambda_l / (lambda_l - lambda_r) * (pm.math.exp(-lambda_r * value[i]) - pm.math.exp(-lambda_l * value[i]))

        ans = pm.math.log(J)
        return ans

    def sample_marginal_posterior(self, Ns):

        model = pm.Model()
        with model:
            beta_l = pm.Exponential('beta_l', scale=self.beta_l_prior_scale, transform=None)
            beta_r = pm.Exponential('beta_r', scale=self.beta_l_prior_scale, transform=None)
            pm.CustomDist('marginal_model_log_likelihood', beta_l, beta_r, logp=self.__marginal_model_log_likelihood, observed=self.t_samples)
            step = pm.Metropolis()
            initvals = {'beta_l' : 10, 'beta_r' : 1}
            trace = pm.sample(Ns, step=step, initvals=initvals, return_inferencedata=False)
        return trace
