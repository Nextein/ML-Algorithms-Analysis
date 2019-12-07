# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def plot_line(ax,w):
   X = np.zeros((2,2))
   X[0,0] = -5.0
   X[1,0] = 5.0
   X[:,1] = 1.0
   y = w.dot(X.T)
   ax.plot(X[:,0],y)
   
tau = 1.0*np.eye(2)
w_0 = np.zeros((2,1))

n_samples = 100

w_samp = np.random.multivariate_normal(w_0.flatten(),tau,size=n_samples)


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

for i in range(w_samp.shape[0]):
    plot_line(ax, w_samp[i,:])
    
plt.show()
