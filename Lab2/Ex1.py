import random

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
X=0
#Ex1

x1 = stats.expon.rvs(0,scale=1/4, size=10000)
x2 = stats.expon.rvs(0,scale=1/6,size=10000)

if random.randint(0,100) <=60:
    X+=x1*random.randrange(240)
else:
    X+=x2*random.randrange(360)

az.plot_posterior({'x1':x1,'x2':x2,'Minutes':X})

plt.show()

