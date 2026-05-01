"""
Rastrigin test function.

f(x) = A N + sum_{i=0}^{N-1} [x_i^2 - A cos(2 pi x_i)]

Global minimum: f(0,...,0) = 0.
"""

import numpy as np


def rastrigin(N: int, A: float = 10.0):
    """
    Parameters
    ----------
    N : int
        Number of variables (N >= 1).
    A : float
        Oscillation amplitude. Standard choice is A = 10.

    Returns
    -------
    f    : callable  x -> float
    grad : callable  x -> ndarray shape (N,)
    hess : callable  x -> ndarray shape (N, N)
    """
    if N < 1:
        raise ValueError("N must be >= 1.")

    two_pi = 2.0 * np.pi
    hess_scale = 4.0 * np.pi ** 2 * A

    def f(x):
        x = np.asarray(x, dtype=float)
        return float(A * N + np.sum(x ** 2 - A * np.cos(two_pi * x)))

    def grad(x):
        x = np.asarray(x, dtype=float)
        return 2.0 * x + two_pi * A * np.sin(two_pi * x)

    def hess(x):
        x = np.asarray(x, dtype=float)
        diagonal = 2.0 + hess_scale * np.cos(two_pi * x)
        return np.diag(diagonal)

    return f, grad, hess
