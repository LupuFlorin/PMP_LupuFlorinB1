import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy.stats as stats


def func():

    N = 10000
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    outside = np.invert(inside)
    plt.figure(figsize=(8, 8))
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error:4.3f}', alpha=0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=1, frameon=True, framealpha=0.9)
    return x,y,inside,outside,pi,error

def func_N(N):

    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    outside = np.invert(inside)
    plt.figure(figsize=(8, 8))
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error:4.3f}', alpha=0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=1, frameon=True, framealpha=0.9)
    return x,y,inside,outside,pi,error

if __name__ == '__main__':

    figure, axis = plt.subplots(2, 2)
    for i in range (0,2):
        for j in range(0,2):
            x,y,inside,outside,pi,error=func()
            axis[i,j].plot(x[inside], y[inside], 'b.')
            axis[i,j].plot(x[outside], y[outside], 'r.')
            axis[i,j].plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error:4.3f}', alpha=0)
            axis[i,j].axis('square')
            axis[i, j].legend(loc=1, frameon=True, framealpha=0.9)


    N=[100,1000,10000]
    for n in N:
        x, y, inside, outside, pi, error = func_N(n)
    plt.show()