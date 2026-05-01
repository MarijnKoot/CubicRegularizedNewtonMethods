"""
Cubic-norm test function.

f(x) = (1/3) ||x||^3

Convex (PSD Hessian everywhere) but not strongly convex — the Hessian is
singular at x = 0. Global minimum is f(0) = 0.

Gradient:
    grad f(x) = ||x|| * x

Hessian:
    H(x) = ||x|| * I + (1/||x||) * x xᵀ   for x ≠ 0
    H(0) = 0

Hessian Lipschitz constant:
    The derivative of H is constant (linear in x with operator norm 2),
    so L3 = 2 globally.
"""

import numpy as np

L3 = 2.0


def cubic_norm(n: int = 2):
    """
    Parameters
    ----------
    n : number of variables

    Returns
    -------
    f, grad, hess, L3  — callables and the exact Hessian Lipschitz constant
    """

    def f(x):
        x = np.asarray(x, dtype=float)
        return (1.0 / 3.0) * float(np.linalg.norm(x)) ** 3

    def grad(x):
        x = np.asarray(x, dtype=float)
        r = float(np.linalg.norm(x))
        return r * x

    def hess(x):
        x = np.asarray(x, dtype=float)
        r = float(np.linalg.norm(x))
        if r == 0.0:
            return np.zeros((n, n))
        return r * np.eye(n) + (1.0 / r) * np.outer(x, x)

    return f, grad, hess, L3
