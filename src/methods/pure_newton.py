"""
Pure Newton method — no line search, no regularization.

Takes the full Newton step h = -H(x)^{-1} g(x) at every iteration.
Singular or indefinite Hessians get a small diagonal jitter for solvability,
but no attempt is made to ensure descent or acceptance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.linalg import norm, solve, LinAlgError

Array = np.ndarray


@dataclass
class PureNewtonOptions:
    """
    Hyperparameters for PureNewton.

    tol_grad   Gradient norm stopping tolerance (||∇f|| <= tol_grad).
    tol_step   Step norm stopping tolerance (||h|| <= tol_step).
    max_iter   Maximum number of iterations.
    """
    tol_grad: float = 1e-6
    tol_step: float = 1e-9
    max_iter: int = 200
    verbose: bool = False


class PureNewton:
    """
    Pure Newton method: x_{k+1} = x_k - H(x_k)^{-1} g(x_k).

    No line search, no regularization, no step acceptance test.
    """

    def __init__(
        self,
        f: Callable[[Array], float],
        grad: Callable[[Array], Array],
        hess: Callable[[Array], Array],
        options: Optional[PureNewtonOptions] = None,
    ):
        """
        Parameters
        ----------
        f       : Callable[[array], float]  — objective function
        grad    : Callable[[array], array]  — gradient of f
        hess    : Callable[[array], array]  — Hessian of f (returns n×n array)
        options : PureNewtonOptions, optional — algorithm hyper-parameters;
                  defaults used when omitted (tol_grad=1e-6, tol_step=1e-9, max_iter=200)
        """
        self.f = f
        self.grad = grad
        self.hess = hess
        self.opt = options or PureNewtonOptions()
        self.log: List[Dict] = []
        self.runtime: Optional[float] = None
        self.termination_reason: Optional[str] = None

    def run(self, x0: Array) -> Array:
        """
        Run the solver from initial point x0.

        Parameters
        ----------
        x0 : array-like, shape (n,)
            Starting point.

        Returns
        -------
        x : ndarray, shape (n,)
            Approximate minimizer at termination.

        Side effects
        ------------
        Populates self.log, self.runtime, self.termination_reason.
        """
        x = np.asarray(x0, dtype=float).reshape(-1)
        n = x.size
        self.log = []
        start = time.perf_counter()
        reason = None

        for k in range(self.opt.max_iter):
            g = np.asarray(self.grad(x), dtype=float).reshape(-1)
            H = np.asarray(self.hess(x), dtype=float)
            H = 0.5 * (H + H.T)
            g_norm = float(norm(g))
            f_x = float(self.f(x))

            if self.opt.verbose:
                print(f"iter {k:4d}  f={f_x:.6e}  ||g||={g_norm:.3e}")

            if g_norm <= self.opt.tol_grad:
                reason = "grad_tol"
                break

            # Solve H h = -g; add small diagonal jitter if H is singular.
            # jitter = 1e-12 * (1 + ||H||_inf) keeps the perturbation proportional
            # to the matrix scale, matching the same pattern used in NCR and ARC's
            # solve_cubic_subproblem.
            try:
                h = solve(H, -g)
            except LinAlgError:
                jitter = 1e-12 * (1.0 + np.linalg.norm(H, np.inf))
                h = solve(H + jitter * np.eye(n), -g)

            h_norm = float(norm(h))

            self.log.append({
                "iter": k,
                "f": f_x,
                "g": g.copy(),
                "H": H.copy(),
                "grad_norm": g_norm,
                "step_norm": h_norm,
                "h": h.copy(),
                "accepted": True,
                "sigma": 0.0,
                "x": x.copy(),
            })

            x = x + h

            if h_norm <= self.opt.tol_step:
                reason = "step_tol"
                break

        else:
            reason = "max_iter"

        self.runtime = time.perf_counter() - start
        self.termination_reason = reason
        return x
