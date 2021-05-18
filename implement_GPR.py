'''
Implemantation of Gaussian process regression from the book
"Gaussian process Regression for machine learning, Rasmussen & Williams"
algorithm is in page 19
The code is implemented via the lecture of Nando De Freitas'
Youtube lectures
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from autograd import value_and_grad


def SE_kernel(a, b, hyp):
    ''' Squared exponential kernel (RBF)'''
    # Kernel parameters
    l = hyp[0]  # Implemented from Rasmussen & Williams
    sigma_n = hyp[1]
    # sq_dist = np.sum(a**2, axis=1).reshape(-1, 1) + np.sum(b**2, axis=1) - 2*np.dot(a, b.T)
    diffs = np.expand_dims(a/l, 1) - np.expand_dims(b/l, 0)
    # return sigma_n**2 * np.exp(-0.5 * sq_dist / (l**2))
    return sigma_n * np.exp(-0.5 * np.sum(diffs**2, axis=2))


def f(x):
    ''' True function '''
    return (np.sin(0.9*x)).flatten()   # flatten collapses the array into one dimension


N = 20      # Number of test points
noise = 0   # noise
s = 1e-5    # noise variance

# Training set
X = np.random.uniform(-5, 5, size=(N, 1))
y = f(X) + noise*np.random.randn(N)
# Implement these hyperparameters for now
# It will be changed with log marginal likelihood


# Test set
nn = 200
X_star = np.linspace(-5, 5, nn).reshape(-1, 1)
y_star = f(X_star)

# Training

init_hyp = np.array([1, 1])
l = init_hyp[0]


def LML(X, y, init_hyp):
    N = y.shape[0]
    # l = init_hyp[0]
    # sigman = init_hyp[1]

    K = SE_kernel(X, X, init_hyp)
    L = np.linalg.cholesky(K + np.eye(N)*s)
    alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))

    LML = -0.5 * np.matmul(np.transpose(y), alpha) - \
        np.sum(np.log(np.diag(L))) - 0.5 * N * np.log(2.0*np.pi)
    return LML


# LML(X, y, init_hyp)
# hyp = minimize(value_and_grad(LML), init_hyp, args=(X, y),
#                method='L-BFGS-B', jac=True)
# hyp
# Prediction of mean
K = SE_kernel(X, X, init_hyp)
L = np.linalg.cholesky(K + np.eye(N)*s)
alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))
k_star = SE_kernel(X, X_star, init_hyp)

mu_star = np.matmul(np.transpose(k_star), alpha)

# Prediction of variance
K_ss = SE_kernel(X_star, X_star, init_hyp)
v = np.linalg.solve(L, k_star)
beta = np.linalg.solve(np.transpose(L), v)

var_f_star = K_ss - np.matmul(np.transpose(k_star), beta)
y_var = np.abs(np.diag(var_f_star))

# Plotting predictions with given hyperparameters
# No Training is performed

plt.figure(1, figsize=(10, 8))
plt.plot(X_star, y_star, 'b-', label="Exact")
plt.plot(X_star, mu_star, 'r--', label="Prediction")

lower_bound = mu_star - 2.0*np.sqrt(y_var)
upper_bound = mu_star + 2.0*np.sqrt(y_var)


plt.gca().fill_between(X_star.flatten(), lower_bound.flatten(), upper_bound.flatten(),
                       facecolor='gray', alpha=0.3, label="2 std band")

plt.plot(X, y, 'bo', markersize=12, alpha=0.5, label="Data")
plt.legend(frameon=False, loc='upper left')
ax = plt.gca()
ax.set_xlim([-5, 5])
ax.set_ylim([-1.5, 1.5])
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Gaussian Process Regression with l = {}'.format(l))
plt.show()
