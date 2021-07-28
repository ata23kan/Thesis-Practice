'''
Class
'''

import numpy as np
from scipy.optimize import minimize
from autograd import value_and_grad

class Multifidelity:
    def __init__(self, X_L, y_L, X_H, y_H):
        self.D = X_H.shape[1]
        self.X_L = X_L
        self.y_L = y_L
        self.X_H = X_H
        self.y_H = y_H

        self.hyp = self.init_params()
        print("Total number of parameters: %d" % (self.hyp.shape[0]))

        self.jitter = 1e-8

    # Initialize hyper-parameters
    def init_params(self):
        hyp = np.log(np.ones(self.D+1))
        self.idx_theta_L = np.arange(hyp.shape[0])

        hyp = np.concatenate([hyp, np.log(np.ones(self.D+1))])
        self.idx_theta_H = np.arange(self.idx_theta_L[-1]+1, hyp.shape[0])

        rho = np.array([1.0])
        logsigma_n = np.array([-4.0, -4.0])
        hyp = np.concatenate([hyp, rho, logsigma_n])
        return hyp

    # A simple vectorized rbf kernel
    def kernel(self,x,xp,hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims(x /lengthscales, 1) - \
                np.expand_dims(xp/lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

    # Computes the negative log-marginal likelihood
    def likelihood(self, hyp):
        X_L = self.X_L
        y_L = self.y_L
        X_H = self.X_H
        y_H = self.y_H

        y = np.vstack((y_L,y_H))

        NL = y_L.shape[0]
        NH = y_H.shape[0]
        N = y.shape[0]

        rho = hyp[-3]
        logsigma_n_L = hyp[-2]
        logsigma_n_H = hyp[-1]
        sigma_n_L = np.exp(logsigma_n_L)
        sigma_n_H = np.exp(logsigma_n_H)

        theta_L = hyp[self.idx_theta_L]
        theta_H = hyp[self.idx_theta_H]

        K_LL = self.kernel(X_L, X_L, theta_L) + np.eye(NL)*sigma_n_L
        K_LH = rho*self.kernel(X_L, X_H, theta_L)
        K_HH = rho**2 * self.kernel(X_H, X_H, theta_L) + \
                        self.kernel(X_H, X_H, theta_H) + np.eye(NH)*sigma_n_H
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = np.linalg.cholesky(K + np.eye(N)*self.jitter)
        self.L = L

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        NLML = 0.5*np.matmul(np.transpose(y),alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5*np.log(2.*np.pi)*N
        return NLML[0,0]

    # Minimizes the negative log-marginal likelihood
    def train(self):
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True,
                          method='L-BFGS-B', callback=self.callback)
        self.hyp = result.x

    # Return posterior mean and variance at a set of test points
    def predict(self,X_star):
        X_L = self.X_L
        y_L = self.y_L
        X_H = self.X_H
        y_H = self.y_H
        L = self.L

        y = np.vstack((y_L,y_H))

        rho = self.hyp[-3]
        theta_L = self.hyp[self.idx_theta_L]
        theta_H = self.hyp[self.idx_theta_H]

        psi1 = rho*self.kernel(X_star, X_L, theta_L)
        psi2 = rho**2 * self.kernel(X_star, X_H, theta_L) + \
                        self.kernel(X_star, X_H, theta_H)
        psi = np.hstack((psi1,psi2))

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = rho**2 * self.kernel(X_star, X_star, theta_L) + \
                     self.kernel(X_star, X_star, theta_H) - np.matmul(psi,beta)

        return pred_u_star, var_u_star

    #  Prints the negative log-marginal likelihood at each training step
    def callback(self,params):
        print("Log likelihood {}".format(self.likelihood(params)))
