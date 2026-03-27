import numpy as np

# Task 2

epsilon = 0.1
x = np.array([np.linspace(0,25)])

def delta(x, epsilon):
    fac = 1/(2*np.pi*epsilon**2)
    e = np.exp(-(np.dot(x.T, x))/(2*epsilon**2))
    return fac * e

def c(x,t,N):
    return 1/N*delta(x,epsilon)*np.sum(x-)