import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from ricker import rickerwavelet
f = 8 #frequency in hertz
tn = 500 #wavelet length in ms
dt = 4 #sampling rate in ms
t,w1  = rickerwavelet(f,tn,dt)
h = hilbert(w1)
deg = 121  #wavelet rotation
theta = deg*np.pi/180
w2 = np.cos(theta)*h.real-np.sin(theta)*h.imag
plt.plot(t,w1,'r', label = 'Zero Phase')
plt.plot(t,w2, 'b', label = 'Rotated %s deg' %deg)
plt.legend()
plt.show()


