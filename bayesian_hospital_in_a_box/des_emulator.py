from typing import Hashable

import numpy as np
import salabim as sim
from bayesian_hospital_in_a_box import likelihood
from matplotlib import pyplot as plt

sim.yieldless(False)


class Specimen(sim.Component):
    def __init__(self,
                 t_store: float,
                 n: int,
                 store: sim.Store,
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
        self.t_store = t_store
        self.n = n
        self.store = store
        self.log = log
        self.log[self.name()] = {}
        self.log[self.name()]['n_times'] = n

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

    def action_work(self):
        if self.n == 0:
            raise ValueError('Specimen has completed all work')
        self.n -= 1

    def work_left(self) -> bool:
        return self.n > 0

    def process(self):
        self.timestamp('time_collected')
        yield self.hold(self.t_store)
        self.timestamp('time_received', normalise=True)
        self.enter(self.store)
        # # yield self.hold(sim.Exponential(rate=self.lambda_r).sample())
        # self.timestamp(event_name='time_received', normalise=True)
        # for _ in range(self.n_l_max):
        #     self.enter(self.store)
        #     yield self.hold(sim.Exponential(rate=self.lambda_l).sample())
        #     self.log[self.name()]['n_times'] += 1
        #     if sim.Uniform(0, 1).sample() < self.pc0:
        #         break
        # self.timestamp('time_completed', normalise=True)

    def culture(self):
        yield self.hold(self.t_store)
        self.enter(self.store)


class Technician(sim.Component):
    def __init__(self, beta, store, t_process):
        super().__init__()
        self.beta = beta
        self.store = store
        self.t_process = t_process

    def process(self):
        while True:
            specimen = yield self.from_store(self.store)
            if specimen is not None:
                yield self.hold(self.t_process)
                specimen.action_work()
                if specimen.work_left():
                    specimen.activate(process="culture")
                else:
                    specimen.timestamp('time_completed', normalise=True)
            wait_time = sim.Exponential(rate=self.beta).sample()
            yield self.hold(wait_time)


def simulate_des(run_until: float,
                 n: int = 1,
                 n_specimens: int = 100,
                 freq: float = 5,
                 t_store: float = 20,
                 t_process: float = 20,
                 beta: float = 1/60,
                 seed: Hashable = "*",
                 print_trace: bool = False) -> dict[str, dict[str, float]]:
    """
    Description
    -----------
        Run a simulation of the DES model.

    Parameters
    ----------
        - theta: model parameters (lambda_r, lambda_l)
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

    Example
    -------
        simulate_des((1/(60*2), 1/(60*8)),
                     0.2,
                     3,
                     10000,
                     seed=42,
                     print_trace=False)
    """
    log: dict = {}
    env = sim.Environment(random_seed=seed,
                          trace=print_trace)
    store = sim.Store('store')
    tech = Technician(beta=beta, store=store, t_process=t_process)
    sim.ComponentGenerator(Specimen,
                           store=store,
                           n=n,
                           number=n_specimens,
                           t_store=t_store,
                           iat=freq,
                           log=log)
    env.run(till=run_until)
    return log


def save_log_csv(log: dict[str, dict[str, float]],
                 filename: str) -> None:
    """
    Description
    -----------
        Save a dictionary of event logs to a CSV file.

    Parameters
    ----------
        - log : dictionary of results
        - filename : name of the file to save the results to

    Example
    -------
        sim_log = simulate_des((1/(60*2), 1/(60*8)),
                            0.2,
                            3,
                            10000,
                            seed=42,
                            print_trace=False)
        save_log_csv(sim_log, 'results.csv')
    """

    import csv
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['specimen', 'time_collected', 'time_received', 'time_completed'])
        for specimen, results in log.items():
            writer.writerow([specimen, results['time_collected'],
                             results['time_received'],
                             results['time_completed']])


def simulated_histogram(log: dict[str, dict[str, float]],
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
        p_t[i] = np.exp(likelihood.logp(theta=theta, t=t, pc0=pc0, N_l_max=n_l_max))

    t_bin_values = t_bin_values / np.sum(t_bin_values)
    p_t = p_t / np.sum(p_t)

    fig, ax = plt.subplots()
    ax.plot(t_bin_centres / 60, t_bin_values, 'black', label='Histogram results')
    ax.plot(t_bin_centres / 60, p_t, 'red', label='Monte Carlo estimate')
    ax.set_xlabel('Hours')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    sim_log = simulate_des(run_until=100000,
                           n=3,
                           n_specimens=10,
                           freq=30,
                           beta=1/60,
                           t_process=20,
                           seed=42,
                           print_trace=False)
    save_log_csv(sim_log, 'results.csv')
