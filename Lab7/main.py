import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv(r'C:\Users\flori\PycharmProjects\Lab7PMP\Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0, 0].scatter(speed, price, alpha=0.6)
    axes[0, 1].scatter(hardDrive, price, alpha=0.6)
    axes[1, 0].scatter(ram, price, alpha=0.6)
    axes[1, 1].scatter(premium, price, alpha=0.6)
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].set_xlabel("Speed")
    axes[0, 1].set_xlabel("HardDrive")
    axes[1, 0].set_xlabel("Ram")
    axes[1, 1].set_xlabel("Premium")
    plt.show()
    plt.savefig('price_correlations.png')

    model =pm.Model()

    with model:
        a = pm.Normal('a', mu=0, sd=10)

        bSpeed = pm.Normal('bSpeed',mu=0, sd=10)
        bHard = pm.Normal('bHard',mu=0,sd=10)

        sigma = pm.HalfNormal('sigma', sd=1)

        mu = pm.Deterministic('mu', a + bSpeed * speed + bHard * np.log(hardDrive))

        price_like = pm.Normal('price_like', mu=mu, sd=sigma, observed=price)

        trace = pm.sample(20000, tune=20000, cores=4)

    a_mean = trace['a'].mean().item()
    bSpeed_mean = trace['bSpeed'].mean().item()
    bHard_mean = trace['bHard'].mean().item()

    ppc = pm.sample_posterior_predictive(trace,samples=100, model=model)



