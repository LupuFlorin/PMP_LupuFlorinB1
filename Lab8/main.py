import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('Admission.csv')

    admission = data['Admission'].values
    GRE = data['GRE'].values
    GPA = data['GPA'].values

    model = pm.Model()

    with model:

        badmission = pm.Normal('badmission',mu=0,sd=10)
        bGRE = pm.Normal('bGRE',mu=0,sd=10)
        bGPA = pm.Normal('bGPA', mu=0, sd=10)

        sigma = pm.HalfNormal('sigma', sd=1)

        mu = pm.Logistic('mu',badmission + bGRE * GRE + bGPA * GPA)

        y_1 = pm.Bernoulli('y_1', p=mu)

        admission_like = pm.Normal('admission_like', mu=mu, sd=sigma, observed=admission)

        trace = pm.sample(20000, tune=20000, cores=4)

    badmission_mean = trace['badmission'].mean().item()
    bGRE_mean = trace['bGRE'].mean().item()
    bGPA_mean = trace['bGPA'].mean().item()

    ppc = pm.sample_posterior_predictive(trace, samples=100, model=model)

    plt.plot(admission, badmission_mean + bGRE_mean * GRE + bGPA_mean * GPA, 'r')
    sig = az.plot_hdi(admission, ppc['admission_like'], hdi_prob=0.94, color='k')
    plt.xlabel('Admission')
    plt.ylabel('', rotation=0)
    plt.savefig('Medie.png')