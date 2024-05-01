from bayesian_hospital_in_a_box.des import simulate_des
from bayesian_hospital_in_a_box.des import simulated_histogram


def test_simulate_des():
    simulate_des(1/(60*2),
                 1/(60*8),
                 0.2,
                 3,
                 10000,
                 seed=42,
                 print_trace=False)


def test_simulated_histogram(plots=False):
    log = simulate_des(1/(60*2),
                       1/(60*8),
                       0.2,
                       3,
                       10000,
                       seed=42,
                       print_trace=False)
    if plots:
        simulated_histogram(log, [1/(60*2), 1/(60*8)], 0.2, 3)
