"""
Generalised Rosenbrock test function.

f(x) = sum_{i=0}^{N-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

Global minimum: f(1,...,1) = 0.
"""

import numpy as np


def rosenbrock(N: int):
    """
    Parameters
    ----------
    N : int
        Number of variables (N >= 2).

    Returns
    -------
    f    : callable  x -> float
    grad : callable  x -> ndarray shape (N,)
    hess : callable  x -> ndarray shape (N, N)
    """
    if N < 2:
        raise ValueError("N must be >= 2.")

    def f(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def grad(x):
        x = np.asarray(x, dtype=float)
        g = np.zeros(N)
        # contribution from term i (0 <= i <= N-2): affects x[i] and x[i+1]
        d = x[1:] - x[:-1] ** 2          # shape (N-1,)
        g[:-1] += -400.0 * x[:-1] * d + 2.0 * (x[:-1] - 1.0)
        g[1:]  += 200.0 * d
        return g

    def hess(x):
        x = np.asarray(x, dtype=float)
        H = np.zeros((N, N))
        d = x[1:] - x[:-1] ** 2          # shape (N-1,)
        # diagonal
        H[np.arange(N - 1), np.arange(N - 1)] += 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
        H[np.arange(1, N), np.arange(1, N)]   += 200.0
        # super- and sub-diagonal
        off = -400.0 * x[:-1]
        H[np.arange(N - 1), np.arange(1, N)] = off
        H[np.arange(1, N), np.arange(N - 1)] = off
        return H

    return f, grad, hess
