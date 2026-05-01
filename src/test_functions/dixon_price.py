"""
Dixon-Price test function.

f(x) = (x_0 - 1)^2 + sum_{i=1}^{N-1} (i + 1) * (2 x_i^2 - x_{i-1})^2

Global minimum:
    x_0 = 1
    x_i = sqrt(x_{i-1} / 2),  i >= 1
with f(x*) = 0.
"""

import numpy as np


def dixon_price(N: int):
    """
    Parameters
    ----------
    N : int
        Number of variables (N >= 1).

    Returns
    -------
    f    : callable  x -> float
    grad : callable  x -> ndarray shape (N,)
    hess : callable  x -> ndarray shape (N, N)
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    weights = np.arange(2.0, N + 1.0)

    def f(x):
        x = np.asarray(x, dtype=float)
        if N == 1:
            return float((x[0] - 1.0) ** 2)
        residual = 2.0 * x[1:] ** 2 - x[:-1]
        return float((x[0] - 1.0) ** 2 + np.sum(weights * residual ** 2))

    def grad(x):
        x = np.asarray(x, dtype=float)
        g = np.zeros(N)
        g[0] = 2.0 * (x[0] - 1.0)
        if N == 1:
            return g

        residual = 2.0 * x[1:] ** 2 - x[:-1]
        g[:-1] += -2.0 * weights * residual
        g[1:] += 8.0 * weights * x[1:] * residual
        return g

    def hess(x):
        x = np.asarray(x, dtype=float)
        H = np.zeros((N, N))
        H[0, 0] = 2.0
        if N == 1:
            return H

        H[:-1, :-1] += np.diag(2.0 * weights)
        H[1:, 1:] += np.diag(8.0 * weights * (6.0 * x[1:] ** 2 - x[:-1]))
        off_diag = -8.0 * weights * x[1:]
        idx = np.arange(N - 1)
        H[idx, idx + 1] = off_diag
        H[idx + 1, idx] = off_diag
        return H

    return f, grad, hess
