"""
Log-Sum-Exp (LSE) Test Function

f_mu(x) = mu * log( sum_{i=1}^m exp((a_i^T x - b_i) / mu) )

This function is quasi-self-concordant with parameter M = 2 / mu.
"""

import numpy as np


def logsumexp(a, b, mu):
    """
    Parameters
    ----------
    a : ndarray, shape (m, n)
        Data matrix (rows are a_i^T).
    b : ndarray, shape (m,)
        Offset vector.
    mu : float
        Smoothing parameter.

    Returns
    -------
    f : callable
    grad : callable
    hess : callable
    M : float
        Quasi-self-concordance parameter.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = a.shape

    def _weights(x):
        z = (a @ x - b) / mu
        z_shift = z - np.max(z)  # numerical stability
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z)

    def f(x):
        z = (a @ x - b) / mu
        z_max = np.max(z)
        z_shift = z - z_max
        return mu * (np.log(np.sum(np.exp(z_shift))) + z_max)

    def grad(x):
        w = _weights(x)
        return a.T @ w

    def hess(x):
        w = _weights(x)
        A_bar = a.T @ w                      # gradient
        H = np.zeros((n, n))
        for i in range(m):
            ai = a[i][:, None]
            H += w[i] * (ai @ ai.T)
        return H - np.outer(A_bar, A_bar)

    M = 2.0 / mu
    return f, grad, hess, M
