"""
Matrix Scaling and Matrix Balancing Test Functions
"""

import numpy as np


def matrix_scaling(A):
    """
    f(x, y) = sum_{i,j} A_ij * exp(x_i - y_j)
    """

    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    def f(z):
        x = z[:n]
        y = z[n:]
        return np.sum(A * np.exp(x[:, None] - y[None, :]))

    def grad(z):
        x = z[:n]
        y = z[n:]
        E = A * np.exp(x[:, None] - y[None, :])

        gx = np.sum(E, axis=1)
        gy = -np.sum(E, axis=0)
        return np.concatenate([gx, gy])

    def hess(z):
        x = z[:n]
        y = z[n:]
        E = A * np.exp(x[:, None] - y[None, :])

        H = np.zeros((2*n, 2*n))
        H[:n, :n] = np.diag(np.sum(E, axis=1))
        H[n:, n:] = np.diag(np.sum(E, axis=0))
        H[:n, n:] = -E
        H[n:, :n] = -E.T
        return H

    return f, grad, hess


def matrix_balancing(A):
    """
    f(x) = sum_{i,j} A_ij * exp(x_i - x_j)
    """

    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    def f(x):
        return np.sum(A * np.exp(x[:, None] - x[None, :]))

    def grad(x):
        E = A * np.exp(x[:, None] - x[None, :])
        return np.sum(E, axis=1) - np.sum(E, axis=0)

    def hess(x):
        E = A * np.exp(x[:, None] - x[None, :])
        H = np.diag(np.sum(E, axis=1) + np.sum(E, axis=0))
        H -= E + E.T
        return H

    return f, grad, hess
