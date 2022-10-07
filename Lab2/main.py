import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

#Ex1

x1 = stats.expon.rvs(0,scale=1/4, size=10000)
x2 = stats.expon.rvs(0,scale=1/6,size=10000)
X = x1+x2 - (x1*40/100+x2*60/100)

az.plot_posterior({'x1':x1,'x2':x2,'X':X})
plt.show()

