"""
Cubic-regular smooth polynomial test function.

f(x) = (1/2) x^T Q x + (gamma/3) ||x||^3

where Q is symmetric positive semidefinite and gamma > 0.

This function mirrors the structure of the CRN cubic model, providing a
near-ideal geometry for the method.

Gradient:
    grad f(x) = Q x + gamma * ||x|| * x

Hessian (||x|| > 0):
    H(x) = Q + gamma * (||x|| * I + x x^T / ||x||)

Hessian at x = 0:
    H(0) = Q
"""

import numpy as np


def cubic_poly(Q, gamma: float = 1.0):
    """
    Parameters
    ----------
    Q     : (n, n) symmetric PSD matrix
    gamma : float, regularization strength (> 0)

    Returns
    -------
    f, grad, hess  — each callable x -> scalar / ndarray
    """
    Q = np.asarray(Q, dtype=float)
    gamma = float(gamma)
    n = Q.shape[0]

    def f(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * float(x @ (Q @ x)) + (gamma / 3.0) * float(np.linalg.norm(x) ** 3)

    def grad(x):
        x = np.asarray(x, dtype=float)
        r = float(np.linalg.norm(x))
        return Q @ x + gamma * r * x

    def hess(x):
        x = np.asarray(x, dtype=float)
        r = float(np.linalg.norm(x))
        if r == 0.0:
            return Q.copy()
        return Q + gamma * (r * np.eye(n) + np.outer(x, x) / r)

    return f, grad, hess, 2.0 * gamma  # L3 = 2*gamma (Lemma 5, Nesterov 2008)
