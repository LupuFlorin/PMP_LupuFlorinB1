import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

az.style.use('arviz-darkgrid')

if __name__ == "__main__":

    #1a
    dummy_data = np.loadtxt(r'C:\Users\flori\PycharmProjects\Lab10PMP\date.csv')

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]

    order = 3

    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1c = (x_1s - x_1s.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    y_1c = (y_1s - y_1s.mean()) / y_1s.std()

    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)


    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)  #sd=10
        #β = pm.Normal('β', mu=0, sd=100, shape=order) #sd=100
        #β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order) #sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_c:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        #β = pm.Normal('β', mu=0, sd=100, shape=order)
        #β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.mult_dot([β, x_1s, x_1c])
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1p)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    waic_l = az.waic(idata_l, scale="deviance")
    loo_l = az.loo(idata_l, scale="deviance")

    waic_p = az.waic(idata_p, scale="deviance")
    loo_p = az.loo(idata_p, scale="deviance")

    waic_c = az.waic(idata_c, scale="deviance")
    loo_c = az.loo(idata_c, scale="deviance")

    cmp_df_waic = az.compare({'model_l': idata_l, 'model_p': idata_p, 'model_c':idata_c},
                        method='BB-pseudo-BMA', ic="waic", scale="deviance")

    cmp_df_loo = az.compare({'model_l': idata_l, 'model_p': idata_p, 'model_c':idata_c},
                        method='BB-pseudo-BMA', ic="loo", scale="deviance")

