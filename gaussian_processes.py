VISUAL = True
import numpy as np
import matplotlib.pyplot as plt
#Function space as Eucidean space:
from scipy.spatial.distance import cdist
# MEAN = 0
# COVARIANCE = K in terms of inputs (x1,x2):
def rbf_kernel(x1,x2,varSigma,lengthscale):
    if x2 is None:
        d = cdist(x1,x1)
    else:
        d = cdist(x1,x2)
    K = varSigma*np.exp(-np.power(d,2)/lengthscale)
    return K

def lin_kernel(x1,x2,varSigma):
    if x2 is None:
        return varSigma*x1.dot(x1.T)
    else:
        return varSigma*x1.dot(x2.T)
def white_kernel(x1,x2,varSigma):
    if x2 is None:
        return varSigma*np.eye(x1.shape[0])
    else:
        return np.zeros(x1.shape[0], x2.shape[0])

def periodic_kernel(x1,x2,varSigma,period,lengthscale):
    if x2 is None:
        d = cdist(x1,x1)
    else:
        d = cdist(x1,x2)
    return varSigma*np.exp(-(2*np.sin((np.pi/period)*np.sqrt(d))**2)/lengthscale**2)


# Sample from prior - select marginal as index set
x = np.linspace(-6,6,200).reshape(-1,1)

K = periodic_kernel(x,None,2.0,5,1) +white_kernel(x,None,1.0)
mu = np.zeros(x.shape).reshape(-1,1)
# 20 samples from Gaussian distr
f = np.random.multivariate_normal(mu.flatten(),K,20)
if VISUAL:
    fig,ax = plt.subplots(1)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.plot(x,f.T)


    plt.show()