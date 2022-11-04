from multiprocessing import freeze_support

import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm


df = pd.read_csv(r'C:\Users\flori\PycharmProjects\Lab6PMP\data.csv')
if __name__ == '__main__':


    test = df["ppvt"]
    age = df["momage"]

    #plt.plot(test, age, 'o')

    #plt.show()

    basic_model = pm.Model()

    with basic_model:
        alpha = pm.Normal('alpha',mu=0,sigma=10)
        beta = pm.Normal('beta',mu=0,sigma=10)
        epsilon = pm.HalfCauchy('epsilon',5)
        u=pm.Deterministic('u',alpha+beta*test)
        y_pred = pm.Normal('y_pred',mu=u,sd=epsilon,observed=age)

        idata_g =pm.sample(100,tune=100,return_inferencedata=True)

        plt.plot_trace(idata_g)
        plt.show()

