import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()
if __name__ == '__main__':
    with model:
        clienti = pm.Poisson('C', mu=20)
        comanda = pm.Normal('Cmd',mu=1,sigma=0.5)
        gatit = pm.Exponential('G',lam=1/10) #alpha=10
        trace = pm.sample(2000)

        dictionary = {
            'clienti': trace['C'].tolist(),
            'comanda': trace['Cmd'].tolist(),
            'gatit': trace['G'].tolist(),
        }



        pm.traceplot(trace)
        plt.show()
