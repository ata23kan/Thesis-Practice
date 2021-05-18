'''
Implementation of the Gaussian processes regression
of the Scikit-learn package
'''
# %% Necessary packages
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt

# %%  Training
init_hyp = np.array([0.4, 0.1])
output_scale = init_hyp[0]
lengthscale = init_hyp[1]

kernel = output_scale * RBF(length_scale=lengthscale)

N = 10      # Number of test points
noise = 0   # noise
s = 1e-5    # noise variance


def f(x):
    ''' True function '''
    return (np.sin(0.9*x)).flatten()   # flatten collapses the array into one dimension


# Training set
X = np.random.uniform(-5, 5, size=(N, 1))
y = f(X) + noise*np.random.randn(N)

gpr = GaussianProcessRegressor(kernel=kernel, alpha=s, random_state=0,
                               optimizer='fmin_l_bfgs_b', n_restarts_optimizer=2)
gpr.fit(X, y)

# %% testing
# Test set
nn = 200
X_star = np.linspace(-5, 5, nn).reshape(-1, 1)
y_star = f(X_star)

y_pred, sigma = gpr.predict(X_star, return_std=True)


# %% Plotting
plt.figure(figsize=(10, 8))
plt.plot(X_star, y_star, 'b-', label='Exact')
plt.plot(X_star, y_pred, 'r--', label='Prediction')

lower_bound = y_pred - 2.0*sigma
upper_bound = y_pred + 2.0*sigma

plt.gca().fill_between(X_star.flatten(), lower_bound.flatten(), upper_bound.flatten(),
                       facecolor='gray', alpha=0.3, label="2 std band")

plt.plot(X, y, 'bo', markersize=12, alpha=0.5, label="Data")
plt.legend(frameon=True, loc='upper left')
ax = plt.gca()
ax.set_xlim([-5, 5])
ax.set_ylim([-1.5, 1.5])
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.title('Gaussian Process Regression with sklearn package')
plt.show()

# %% Optimized hyperparameters

LML = gpr.log_marginal_likelihood_value_
LML

sigman = gpr.kernel_.theta[0]
l = gpr.kernel_.theta[1]
sigman
l
