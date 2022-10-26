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
        timp = pm.Normal('T',gatit+comanda)
        trace = pm.sample(2000)

        dictionary = {
            'clienti': trace['C'].tolist(),
            'comanda': trace['Cmd'].tolist(),
            'gatit': trace['G'].tolist(),
            'timp': trace['T'].tolist()
        }
        df=pd.DataFrame(dictionary)

        #Valoarea alpha=10 este voloarea maxima pentru care probabiltitatea este 95% ca timpul de servire este mai mic de 15 minute
        p_timp= df[(df['timp'] <=15)].shape[0] / df.shape[0]

        sumtimp=0
        for idx in range(df.shape[0]):
            sumtimp+=df.at[idx,'timp']

        medie= sumtimp /df.shape[0]
        print(p_timp)
        print(medie)

