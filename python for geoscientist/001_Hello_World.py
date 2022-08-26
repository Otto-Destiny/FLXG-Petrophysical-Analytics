import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,360+10,10)
y = np.sin(x*np.pi/180)
plt.plot(x,y)
plt.xlabel('TWT')
plt.ylabel('Amplitude')
plt.show()
