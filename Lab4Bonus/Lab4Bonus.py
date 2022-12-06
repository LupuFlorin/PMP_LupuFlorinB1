import pandas as pd
import numpy as np
import pymc3 as pm
from scipy import stats

if __name__ == '__main__':

    alpha = 2
    nr_case=5
    nr_statii=5
    for x in range(20):
            model = pm.Model()
            with model:

                nr_clienti = pm.Poisson('N', mu=20)
                t_casa = pm.Normal('T_c', mu=1, sd=0.5, shape=50)
                t_gatit = pm.Exponential('T_g', lam=1/alpha, shape=50)
                t_masa = pm.Normal('T_m', mu=10, sd=2, shape=50, initval=0)
                idx = np.arange(50)
                timp = pm.math.switch(nr_clienti > idx, t_casa[idx] / nr_case + t_gatit[idx] / nr_statii + t_masa[idx], 0)
                succes = pm.Deterministic('S', pm.math.prod(pm.math.switch(timp < 15, 1, 0)))
                trace = pm.sample(10000)

            succese = trace['S']
            prob = len(succese[(succese == 1)]) / len(succese)
            if prob>0.95:
                print(nr_case, nr_statii, prob)
                break
            else:
                nr_case+=1
                nr_statii+=1



