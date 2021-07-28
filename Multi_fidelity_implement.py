'''
Implementation of Multi-fidelity Regression
@author: Atakan Aygün
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from GaussianProcesses import Multifidelity
from pyDOE import lhs

def f_H(x):
    return (6.0*x-2.0)**2 * np.sin(12.*x-4.0)

def f_L(x):
    return 0.5*f_H(x) + 10.0*(x-0.5) - 5.0

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)


if __name__ == "__main__":

    N_H = 3
    N_L = 8
    D = 1
    lb = 0.0*np.ones(D)
    ub = 1.0*np.ones(D)
    noise_L = 0.00
    noise_H = 0.00

    Normalize_input_data = 1
    Normalize_output_data = 1

    # Training data
    X_L = lb + (ub-lb)*lhs(D, N_L)
    y_L = f_L(X_L) + noise_L*np.random.randn(N_L,D)

    X_H = lb + (ub-lb)*lhs(D, N_H)
    y_H = f_H(X_H) + noise_H*np.random.randn(N_H,D)

    # Test data
    nn = 200
    X_star = np.linspace(lb, ub, nn)[:,None]
    y_star = f_H(X_star)

     #  Normalize Input Data
    if Normalize_input_data == 1:
        X = np.vstack((X_L,X_H))
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)
        X_L = Normalize(X_L, X_m, X_s)
        X_H = Normalize(X_H, X_m, X_s)
        lb = Normalize(lb, X_m, X_s)
        ub = Normalize(ub, X_m, X_s)
        X_star = Normalize(X_star, X_m, X_s)

    #  Normalize Output Data
    if Normalize_output_data == 1:
        y = np.vstack((y_L,y_H))
        y_m = np.mean(y, axis = 0)
        y_s = np.std(y, axis = 0)
        y_L = Normalize(y_L, y_m, y_s)
        y_H = Normalize(y_H, y_m, y_s)
        y_star = Normalize(y_star, y_m, y_s)

    # Define model
    model = Multifidelity(X_L, y_L, X_H, y_H)

    # Train
    model.train()
