import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file = './WELLS/F02-1_logs.las'
data = np.loadtxt(file,skiprows=35)
data[data==-999.2500]=np.nan
mnemonics = ['DEPTH', 'RHOB', 'DT', 'GR', 'AI', 'AI_rel', 'PHIE']
data = pd.DataFrame(data,columns=mnemonics)
data = data[['DEPTH', 'RHOB', 'DT', 'GR','PHIE']]
data = data.values
rows, cols = 1,3
fig,ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,10), sharey=True)
mnemonics = ['RHOB', 'DT', 'GR','PHIE']
for i in range(cols):
    ax[i].plot(data[:,i+1], data[:,0],linewidth='0.5')
    ax[i].set_ylim(max(data[:, 0]), min(data[:, 0]))
    ax[i].minorticks_on()
    ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='green')
    ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black') #this is a comment
    ax[i].set_title('%s' %mnemonics[i])

y2 = data[:, 3]
y1 = y2*0+50
ax[2].fill_betweenx(data[:, 0], y1,y2, where=(y1>=y2), color='gold', linewidth=0)
ax[2].fill_betweenx(data[:, 0], y1,y2, where=(y1< y2), color='lime', linewidth=0)
plt.subplots_adjust(wspace=0)
plt.show()


