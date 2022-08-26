import numpy as np
def rickerwavelet(f,tn,dt):
    t = np.arange(-tn/2, tn/2+dt, dt)/1000
    w1 = (1-2*np.pi**2*f**2*t**2)*np.exp(-np.pi**2*f**2*t**2)
    return t,w1

