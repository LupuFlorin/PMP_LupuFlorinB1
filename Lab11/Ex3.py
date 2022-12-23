import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy.stats as stats


def metropolis(func, draws=10000):

    trace = np.zeros(draws)
    old_x = func.mean()
    old_prob = func.pmf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pmf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace

if __name__ == '__main__':
    func = stats.betabinom(2, 5, 2)
    trace = metropolis(func=func)
    x = np.linspace(0.01, .99, 100)
    y = func.pmf(x)
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pmf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()