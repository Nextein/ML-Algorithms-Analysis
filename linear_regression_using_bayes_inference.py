# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def plot_line(ax, w):
	#input data
	X = np.zeros((2,2))
	X[0,0] = -5.0
	X[1,0] = 5.0
	X[:,1] = 1.0
	y = w.dot(X.T)
	ax.plot(X[:,0],y)

# Create prior distribution
# Covariance: equal variance for each param and covariance of 0 (params are independent from each other)
sigma = 1.0*np.eye(2)
# Prior is a horizontal line
w_0 = np.zeros((2,1))

n_samples = 100
# With means of 0:
w_samp = np.random.multivariate_normal(w_0.flatten(), sigma,size=n_samples)

#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(111)

#for i in range(w_samp.shape[0]):
#	plot_line(ax, w_samp[i,:])

#plt.show()


'''
The most likely line according to our prior is a horizontal line.
The least likely line is a vertical line.
When defined as a Gaussian, no line has zero probability. (Gaussian becomes 0 at infinity)
'''

def plotdistribution(ax,mu,sigma):
	x = np.linspace(-1.5,1.5,100)
	x1p, x2p = np.meshgrid(x,x)
	pos = np.vstack((x1p.flatten(),x2p.flatten().T))

	pdf.multivariate_normal(mu.flatten(),sigma)
	Z = pdf.pdf(pos)
	Z = Z.reshape(100,100)

	ax.countour(x1p,x2p,Z,5,colors='r', lw=5, alpha = 0.7)

	ax.set_xlabel('w_0')
	ax.set_ylabel('w_1')
	return

X = np.linspace(-5,5,100)
X = np.vstack((X,np.ones((100))))

w_0 = np.array([3,0])
y = w_0.dot(X)




index = np.random.permutation(X.shape[1])

fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(111)

error_precision = 3.33
prior_mu = np.array([1,0]).T
prior_sigma = np.eye(2)

i=20

Xi = X[:,index[:i]].T
Yi = y[index[:i]]

#posterior_mu = (np.linalg.inv(np.linalg.inv(prior_sigma) + error_precision*Xi.T*X)  *  (np.linalg.inv(prior_sigma)*prior_mu+error_precision*X.T*y)
posterior_mu = np.linalg.inv(np.linalg.inv(prior_sigma)+error_precision*Xi.T.dot(Xi)).dot(np.linalg.inv(prior_sigma).dot(prior_mu) + error_precision*Xi.T.dot(Yi))
plot_line(ax2,posterior_mu)
ax2.scatter(Xi[:,0],Yi)

plt.show()