from typing import Hashable

import numpy as np
import salabim as sim
from bayesian_hospital_in_a_box import likelihood
from matplotlib import pyplot as plt

sim.yieldless(False)


class Specimen(sim.Component):
    def __init__(self,
                 lambda_r: float,
                 lambda_l: float,
                 pc0: float,
                 n_l_max: int,
                 log: dict,
                 **kwargs) -> None:
        """
        Description
        -----------
            Specimen class for DES implementation.

        Parameters
        ----------
            - lambda_r : model parameter
            - lambda_l : model parameter
            - pc0 : the probability that C = 0
            - n_l_max : maximum number of cycles through the lab
            - log : dictionary within which to store results
            - **kwargs : additional named arguments to pass to parent sim.Component
        """

        super().__init__(**kwargs)
        self.lambda_r = lambda_r
        self.lambda_l = lambda_l
        self.pc0 = pc0
        self.n_l_max = n_l_max
        self.log = log
        self.log[self.name()] = {}
        self.log[self.name()]['n_times'] = 0

    def timestamp(self,
                  event_name: str,
                  event_time: float | None = None,
                  normalise: bool = False) -> None:
        """
        Description
        -----------
            Method to store timestamps in the self.log dictionary.

        Parameters
        ----------
            - event_name : name of the event
            - event_time : time at which the event occurred (inferred as now if None)
            - normalise : whether to normalise the time (i.e. subtract time_collected)
        """

        _time: float = self.env.now() if not event_time else event_time
        if normalise:
            _time = _time - self.log[self.name()]['time_collected']
        self.log[self.name()][event_name] = event_time if event_time else self.env.now()

    def process(self):
        self.timestamp('time_collected')
        yield self.hold(sim.Exponential(rate=self.lambda_r).sample())
        self.timestamp(event_name='time_received', normalise=True)
        for _ in range(self.n_l_max):
            yield self.hold(sim.Exponential(rate=self.lambda_l).sample())
            self.log[self.name()]['n_times'] += 1
            if sim.Uniform(0, 1).sample() < self.pc0:
                break
        self.timestamp('time_completed', normalise=True)


def simulate_des(lambda_r: float,
                 lambda_l: float,
                 pc0: float,
                 n_l_max: int,
                 n_specimens: int,
                 seed: Hashable = "*",
                 print_trace: bool = False) -> dict[str, dict[str, float]]:
    """
    Description
    -----------
        Run a simulation of the DES model.

    Parameters
    ----------
        - lambda_r : model parameter
        - lambda_l : model parameter
        - pc0 : the probability that C = 0
        - n_l_max : maximum number of cycles through the lab
        - n_specimens : number of specimens to simulate
        - seed : random seed, follow salabim convention, default "*" [random]
        - print_trace : whether to print the trace

    Returns
    -------
        - log dictionary of results:
            - key : specimen name
            - value : dictionary of results (str: float)
    """
    log: dict = {}
    env = sim.Environment(random_seed=seed,
                          trace=print_trace)
    sim.ComponentGenerator(Specimen,
                           lambda_r=lambda_r,
                           lambda_l=lambda_l,
                           pc0=pc0,
                           n_l_max=n_l_max,
                           number=n_specimens,
                           log=log)
    env.run()
    return log


def simulated_histogram(log: dict[str, float],
                        theta,
                        pc0,
                        n_l_max) -> None:
    t_samples = []
    for specimen in log.values():
        t_samples.append(specimen['time_completed'])
    t_bin_values, t_bin_edges = np.histogram(t_samples)
    t_bin_centres = (t_bin_edges[:-1] + t_bin_edges[1:]) / 2

    p_t = np.zeros(len(t_bin_centres))
    for i, t in enumerate(t_bin_centres):
        p_t[i] = likelihood.p(theta=theta, t=t, pc0=pc0, N_l_max=n_l_max)

    t_bin_values = t_bin_values / np.sum(t_bin_values)
    p_t = p_t / np.sum(p_t)

    fig, ax = plt.subplots()
    ax.plot(t_bin_centres / 60, t_bin_values, 'black', label='Histogram results')
    ax.plot(t_bin_centres / 60, p_t, 'red', label='Monte Carlo estiamte')
    ax.set_xlabel('Hours')
    ax.legend()
    plt.show()
