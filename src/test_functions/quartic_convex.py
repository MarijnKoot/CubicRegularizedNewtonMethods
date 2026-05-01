"""
Strongly convex quartic polynomial test function.

f(x) = (1/2) ||x||^2 + (1/4) ||x||^4

Globally strongly convex and smooth; designed to highlight the benefits
of acceleration (ACRN) without interference from non-convex effects.

Gradient:
    grad f(x) = (1 + ||x||^2) x

Hessian:
    H(x) = (1 + ||x||^2) I + 2 x x^T
"""

import numpy as np


def quartic_convex(n: int = 2):
    """
    Parameters
    ----------
    n : int, number of variables

    Returns
    -------
    f, grad, hess  — each callable x -> scalar / ndarray
    """

    def f(x):
        x = np.asarray(x, dtype=float)
        r2 = float(x @ x)
        return 0.5 * r2 + 0.25 * r2 ** 2

    def grad(x):
        x = np.asarray(x, dtype=float)
        r2 = float(x @ x)
        return (1.0 + r2) * x

    def hess(x):
        x = np.asarray(x, dtype=float)
        r2 = float(x @ x)
        return (1.0 + r2) * np.eye(n) + 2.0 * np.outer(x, x)

    return f, grad, hess
