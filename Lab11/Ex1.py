import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy.stats as stats


def posterior_grid(grid_points=200, heads=6, tails=9):

    grid = np.linspace(0, 1, grid_points)
    #prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    #prior = (grid<= 0.5).astype(int)
    #prior = abs(grid - 0.5)
    #prior = (grid<= 0.8).astype(int)
    prior = abs(grid - 0.7)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


if __name__ == '__main__':

    data = np.repeat([0, 1], (10, 3))
    points = 10
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ');
    plt.show()