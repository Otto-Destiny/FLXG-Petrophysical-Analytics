from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from butterworth import  butter_lowpass_filter
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
data = pd.read_csv('well1.csv')
data['VELP']=1000000/data.DT
data = data[['DEPTH', 'RHOB', 'VELP', 'GR','FACIES' ]]
data = data.dropna(how='any')
data['RHOBF'] = butter_lowpass_filter(data.RHOB.values,10,1000/1, order=5)
data['VELPF'] = butter_lowpass_filter(data.VELP.values,10,1000/1, order=5)
data['GRF'] = butter_lowpass_filter(data.GR.values,10,1000/1, order=5)
data = data[['DEPTH', 'RHOBF', 'VELPF', 'GRF','FACIES' ]]
X_train = data.iloc[:,1:4].values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Y_train = data.iloc[:,-1].values
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,Y_train)  #global 'equation' for testing new datasets
X_test = X_train
#### confusion matrix
cor_train = data.corr()
cor_test = data.corr()
ax = sns.heatmap(
    cor_train,
    vmin=-1, vmax=1, center=0,
    cmap='coolwarm',
    square=True,annot = True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=15,
    horizontalalignment='right')
plt.show()
########################
y_pred = model.predict(X_test)
mnemonics = list(data.columns)
data = data.values
rows, cols = 1, 5
fig,ax = plt.subplots(nrows = rows, ncols=cols, figsize=(12,10), sharey=True)
for i in range(cols):
    if i < cols-2:
        ax[i].plot(data[:,i+1],data[:,0],'r', linewidth=0.6)
        ax[i].minorticks_on()
        ax[i].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[i].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax[i].set_ylim(max(data[:, 0]), min(data[:, 0]), 0)
        ax[i].set_title('%s' %mnemonics[i+1])
    elif i==cols-2:
        F = np.vstack((data[:,-1],data[:,-1])).T
        m = ax[i].imshow(F, aspect='auto',cmap='jet', extent=[0,1,max(data[:,0]), min(data[:,0])])
        ax[i].set_title('%s' % mnemonics[i + 1])
    elif i==cols-1:
        F = np.vstack((y_pred,y_pred)).T
        m = ax[i].imshow(F, aspect='auto',cmap='jet', extent=[0,1,max(data[:,0]), min(data[:,0])])
        ax[i].set_title('PREDICTED')
cl = 60
y2 = data[:,3]
y1 = y2*0+cl
ax[2].fill_betweenx(data[:, 0], y1, y2, where=(y1 >= y2), color='gold', linewidth=0)
ax[2].fill_betweenx(data[:, 0], y1, y2, where=(y1 < y2), color='lime', linewidth=0)
plt.subplots_adjust(wspace=0)
plt.show()

