# coding: utf-8
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from pprint import pprint
def posterior(a,b,X):
    a_n = a + X.sum()
    # b_n = b + SUM(1 - x_i) = b + N - SUM(x_i)
    b_n = b + (X.shape[0] - X.sum())
    return beta.pdf(mu_test, a_n, b_n)

def posterior_ab(a,b,X):
    a_n = a + X.sum()
    # b_n = b + SUM(1 - x_i) = b + N - SUM(x_i)
    b_n = b + (X.shape[0] - X.sum())
    return [a_n,b_n]
    
# Generate data
mu = 0.2
N = 500
X = np.random.binomial(1,mu,N)
mu_test = np.linspace(0,1,100)

#prior
a = 10
b = a

#p(mu) = Beta(alpha,beta)
prior_mu = beta.pdf(mu_test,a,b)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)

#plot prior
ax.plot(mu_test, prior_mu, 'g')
ax.fill_between(mu_test, prior_mu, color='green', alpha = 0.3)
ax.set_xlabel('$\mu$')
ax.set_ylabel('$p(\mu|\mathbf{x})$')

#pick random point from data and update posterior with this
index = np.random.permutation(X.shape[0])

N_test = np.linspace(0,N,N);
ax2.set_xlabel('$evidence$')
ax2.set_ylabel('$distance_from_prior$')
mu_prior = beta.mean(a,b)


distance_from_prior = []
for i in range(X.shape[0]):
    y = posterior(a,b,X[:index[i]])
    ax.plot(mu_test,y,color='r',alpha=0.3)
    # Second plot
    mu_posterior = beta.mean(posterior_ab(a,b,X[:index[i]])[0],posterior_ab(a,b,X[:index[i]])[1])
    distance_from_prior.append(mu_posterior - mu_prior)

ax2.plot(N_test, distance_from_prior, color = 'b')

X = np.random.binomial(1,mu,N)
distance_from_prior = []
for i in range(X.shape[0]):
    y = posterior(a,b,X[:index[i]])
    # Second plot
    mu_posterior = beta.mean(posterior_ab(a,b,X[:index[i]])[0],posterior_ab(a,b,X[:index[i]])[1])
    distance_from_prior.append(mu_posterior - mu_prior)
ax3.plot(N_test, distance_from_prior, color = 'b')
X = np.random.binomial(1,mu,N)
distance_from_prior = []
for i in range(X.shape[0]):
    y = posterior(a,b,X[:index[i]])
    # Second plot
    mu_posterior = beta.mean(posterior_ab(a,b,X[:index[i]])[0],posterior_ab(a,b,X[:index[i]])[1])
    var_posterior = beta.var(posterior_ab(a,b,X[:index[i]])[0],posterior_ab(a,b,X[:index[i]])[1])
    distance_from_prior.append(mu_posterior - mu_prior)
    
ax4.plot(N_test, distance_from_prior, color = 'b')
# Final estimation:
y = posterior(a,b,X)
ax.plot(mu_test,y,color='b',linewidth=4.0)


'''
When there is little evidence (little data), posterior will be close to prior.

When our prior is very confident at a place far from the true value,
requires a lot more evidence to move away from prior.

It will quickly (with little evidence) converge at the true value.

The order of the data doesn't seem to make a difference when it comes to the distance to the prior.

It makes sense because we'd expect our posterior belief to be of the same form as our prior belief.
Otherwise the evidence would be dramatically changing our model.
'''



plt.show()