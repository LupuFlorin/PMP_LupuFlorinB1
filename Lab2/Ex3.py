import random

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

#Ex3

recordList=np.chararray((100,10),itemsize=2)

for idx in range(0,100):
    for idx2 in range(0,10):
        flip1 = random.randint(0,1)
        if random.randint(0,10)<=3:
            flip2=0
        else:
            flip2=1

        if flip1==0 and flip2==0:
            recordList[idx][idx2] ='ss'
        elif flip1==1 and flip2==0:
            recordList[idx][idx2] = 'sb'
        elif flip1==0 and flip2==1:
            recordList[idx][idx2] ='bs'
        elif flip1==1 and flip2==1:
            recordList[idx][idx2] ='bb'

print(recordList)