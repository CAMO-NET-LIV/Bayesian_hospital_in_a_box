import numpy as np
import salabim as sim
from bayesian_hospital_in_a_box import likelihood
from matplotlib import pyplot as plt

sim.yieldless(False)


class Specimen(sim.Component):
    def __init__(self, lambda_r, lambda_l, pc0, n_l_max, log, **kwargs):
        super().__init__(**kwargs)
        self.lambda_r = lambda_r
        self.lambda_l = lambda_l
        self.pc0 = pc0
        self.n_l_max = n_l_max
        self.log = log
        self.log[self.name()] = {}
        self.log[self.name()]['n_times'] = 0

    def timestamp(self, event_name, event_time = None, normalise=False):
        _time = self.env.now() if not event_time else event_time
        if normalise:
            _time = _time - self.log[self.name()]['time_collected']
        self.log[self.name()][event_name] = event_time if event_time else self.env.now()

    def process(self):
        self.timestamp('time_collected')
        yield self.hold(sim.Exponential(rate=self.lambda_r).sample())
        self.timestamp('time_received', normalise=True)
        for _ in range(self.n_l_max):
            yield self.hold(sim.Exponential(rate=self.lambda_l).sample())
            self.log[self.name()]['n_times'] += 1
            if sim.Uniform(0, 1).sample() < self.pc0:
                break
        self.timestamp('time_completed', normalise=True)


def simulate_des(lambda_r,
                 lambda_l,
                 pc0,
                 n_l_max,
                 n,
                 seed="*",
                 print_trace=False) -> dict[str, float]:
    log = {}
    env = sim.Environment(random_seed=seed,
                          trace=print_trace)
    generator = sim.ComponentGenerator(Specimen,
                                       lambda_r=lambda_r,
                                       lambda_l=lambda_l,
                                       pc0=pc0,
                                       n_l_max=n_l_max,
                                       number=n,
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
    ax.plot(t_bin_centres/60, t_bin_values, 'black', label='Histogram results')
    ax.plot(t_bin_centres/60, p_t, 'red', label='Monte Carlo estiamte')
    ax.set_xlabel('Hours')
    ax.legend()
    plt.show()
