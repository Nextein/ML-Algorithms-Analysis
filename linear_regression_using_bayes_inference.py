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

fig,ax = plt.subplots(2,2)
fig.set_figheight(10)
fig.set_figwidth(15)

for i in range(w_samp.shape[0]):
	plot_line(ax[0,0], w_samp[i,:])

#plt.show()


'''
The most likely line according to our prior is a horizontal line.
The least likely line is a vertical line.
When defined as a Gaussian, no line has zero probability. (Gaussian becomes 0 at infinity)
'''

def plotdistribution(ax,mu,sigma):
	x = np.linspace(-1.5,1.5,n_samples)
	x1p, x2p = np.meshgrid(x,x)
	pos = np.vstack((x1p.flatten(),x2p.flatten().T))

	pdf = np.random.multivariate_normal(mu.flatten(),sigma)
	Z = pdf.pdf(pos)
	Z = Z.reshape(100,100)

	ax.countour(x1p,x2p,Z,5,colors='r', lw=5, alpha = 0.7)

	ax.set_xlabel('w_0')
	ax.set_ylabel('w_1')
	return

# GENERATE DATA
X = np.linspace(-5,5,n_samples)
X = np.vstack((X,np.ones((n_samples))))
w = np.array([3,0])
sigma = np.eye(2)
#sigma = np.zeros((2,2))
pprint(sigma)
w_samp2 = np.random.multivariate_normal(w.flatten(), sigma,size=n_samples)
# Ys = 100x100 (100 points for each w_samp)
Ys = w_samp2.dot(X)
Y = np.ones((100)).T
#Sample Y:
for i in range(n_samples):
	Y[i] = Ys[i,i]



index = np.random.permutation(X.shape[1])


for i in range(n_samples):
	ax[0,1].scatter(X[0,:], Ys[i,:])
ax[0,1].set_ylabel("Original data")
error_precision = 3.33
prior_mu = np.array([0,0]).T
prior_sigma = np.eye(2)

# Change i to define how much data is used for training
#i=n_samples
i=0

Xi = X[:,index[:i]].T
Yi = Y[index[:i]]

posterior_mu = np.linalg.inv(np.linalg.inv(prior_sigma)+error_precision*Xi.T.dot(Xi)).dot(np.linalg.inv(prior_sigma).dot(prior_mu) + error_precision*Xi.T.dot(Yi))
posterior_sigma = np.linalg.inv(np.linalg.inv(prior_sigma)+error_precision*Xi.T.dot(Xi))
print("prior: {}\noriginal: {}\nposterior: {}".format(prior_mu,w,posterior_mu))
plot_line(ax[1,0],posterior_mu)
ax[1,0].scatter(Xi[:,0],Yi)
plotdistribution has bugs :(
plotdistribution(ax[1,1],posterior_mu,posterior_sigma)

ax[1,0].set_ylabel("posterior")

plt.show()