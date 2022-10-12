import random

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

#Ex2
X=0

x1=stats.gamma.rvs(4,scale=1/3,size=10000)
x2=stats.gamma.rvs(4,scale=1/2,size=10000)
x3=stats.gamma.rvs(5,scale=1/2,size=10000)
x4=stats.gamma.rvs(5,scale=1/3,size=10000)

xi=stats.expon.rvs(0,scale=1/4, size=10000)

if random.randint(0,100) <=25:
    X+=x1*random.randrange(3)+xi
elif random.randint(0,100) <=50:
    X += x2 * random.randrange(2) + xi
elif random.randint(0, 100) <= 80:
    X += x3 * random.randrange(2) + xi
elif random.randint(0,100) <=100:
    X += x4 * random.randrange(3) + xi

az.plot_posterior({'x1':x1,'x2':x2,'x3':x3,'x4':x4,'Miliseconds':X})



plt.show()
