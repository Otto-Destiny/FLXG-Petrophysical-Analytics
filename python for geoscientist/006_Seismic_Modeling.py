import numpy as np
import matplotlib.pyplot as plt
from ricker import rickerwavelet
from scipy.signal import convolve
t,w1  = rickerwavelet(15,500,4)
ns = 250
nt = 51
ref=[]
traces=[]
for i in range(nt):
    R = np.zeros(ns)
    R[50] = 0.8
    R[52+i] =-0.7
    tr = convolve(R,w1, mode='same')
    traces.append(tr)
    ref.append(R)
ref = np.asarray(ref).T
traces = np.asarray(traces).T
plt.subplot(3,1,1)
plt.imshow(traces, aspect='auto', cmap='bwr_r')
plt.xlim(0,50)
plt.subplot(3,1,2)
amp50 = traces[50,:]
plt.plot(amp50)
plt.xlim(0,50)
plt.subplot(3,1,3)
trace9 = traces[:,9]
plt.plot(trace9)
plt.show()


