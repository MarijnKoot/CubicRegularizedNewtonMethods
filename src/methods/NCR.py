"""
Cubic-Regularized Newton (NCR / CRN)

Minimizes a smooth objective f by repeatedly fitting a cubic model
m_k(h) = f(x_k) + g_k^T h + 1/2 h^T H_k h + (sigma_k/3)||h||^3
and taking the global minimizer as the next step candidate. The step is
accepted when f(x_k + h_k) <= m_k(h_k) (majorization / sufficient decrease);
sigma is halved on success and doubled on failure. Under Lipschitz-Hessian
assumptions this gives O(k^{-2}) convergence in gradient norm.

Reference: Nesterov & Polyak (2006). Math. Program. 108, 177–205.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import eigvalsh, norm, solve, LinAlgError

Array = np.ndarray

def solve_cubic_subproblem(g: Array, H: Array, sigma: float) -> Array:
    """
    Global minimizer of m(h) = g^T h + 1/2 h^T H h + (sigma/3)||h||^3
    via the secular equation phi(r)=0 with r >= (-lambda_min(H))/sigma.
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    H = np.asarray(H, dtype=float)
    n = g.size
    if n == 0 or norm(g) == 0.0:
        return np.zeros_like(g)

    H = 0.5 * (H + H.T)  # enforce symmetry
    lam_min = float(eigvalsh(H).min())
    sigma = float(sigma)
    r_low = max(0.0, (-lam_min) / max(sigma, 1e-300))
    I = np.eye(n)

    def safe_solve(mat: Array, rhs: Array) -> Array:
        # If a problem has a singular matrix we run into issues thus, 
        # we solve with some small diagonal jitter to fix these issues
        try:
            return solve(mat, rhs)
        except LinAlgError:
            jitter = 1e-12 * (1.0 + np.linalg.norm(mat, ord=np.inf))
            return solve(mat + jitter * I, rhs)

    def phi(r: float) -> float:
        A = H + (sigma * r) * I
        u = safe_solve(A, g)
        return float(norm(u) - r)

    # phi(r) = ||h(r)|| - r  where h(r) = -(H + sigma*r*I)^{-1} g.
    # The secular equation phi(r)=0 gives the step norm r* that satisfies
    # the cubic optimality condition sigma*r* = lambda_k (the regularizer shift).
    phi_low = phi(r_low)
    if phi_low <= 0.0:
        r_star = r_low
    else:
        # Find root by doubling r_high in (a, +inf , 1e20) until phi <= 0.
        r_high = max(1.0, 2.0 * r_low + 1.0)
        while phi(r_high) > 0.0:
            r_high *= 2.0
            if r_high > 1e20:
                raise RuntimeError("Failed to bracket secular equation root.")
        lo, hi = r_low, r_high
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            val = phi(mid)
            if abs(val) <= 1e-12:
                r_star = mid
                break
            if val > 0.0:
                lo = mid
            else:
                hi = mid
        else:
            r_star = 0.5 * (lo + hi)

    A = H + (sigma * r_star) * I
    h = -safe_solve(A, g)
    return h


@dataclass
class CRNOptions:
    """
    Hyperparameters for CubicNewton (NCR).

    sigma0           Initial regularization parameter (≈ 2·L3 if known; else 0.5).
    sigma_min        Floor for sigma to prevent it shrinking to zero.
    sigma_max        Cap for sigma to prevent unbounded growth.
    tol_grad         Gradient norm stopping tolerance (||∇f|| <= tol_grad).
    tol_step         Step norm stopping tolerance (||h|| <= tol_step).
    max_iter         Maximum number of iterations.
    secular_tol      Bisection convergence tolerance for the secular equation.
    secular_max_iter Max bisection iterations for the secular equation.
    verbose          Print per-iteration progress.
    """
    sigma0: float = 0.5
    sigma_min: float = 5e-13
    sigma_max: float = 5e11
    tol_grad: float = 1e-6
    tol_step: float = 1e-9
    max_iter: int = 200
    secular_tol: float = 1e-12
    secular_max_iter: int = 200
    verbose: bool = False


class CubicNewton:
    """
    Cubic-regularized Newton method using cubic model majorization.

    Implements the basic NCR scheme: solve the cubic subproblem globally at
    each step, accept if the majorization condition holds, and adapt sigma
    by halving/doubling. Unlike ARC it does not use a ratio-based acceptance
    test, and unlike ACRN it requires no Lipschitz constant and works on
    non-convex problems.
    """

    # Algorithm structure (numbered as in the thesis):
    # (1) Initialization: given x0, sigma0 > 0 (safeguard with sigma_min).
    # (2) Local model: m_k(h) = f(x_k) + g_k^T h + 0.5 h^T H_k h + (sigma_k/3)||h||^3.
    # (3) Subproblem: compute h_k = argmin m_k(h); set y_k = x_k + h_k.
    # (4) Acceptance/update: if f(y_k) <= m_k(h_k) (majorization), accept and halve sigma;
    #     else reject, double sigma, and retry. Stop on grad/step tolerances.

    def __init__(
        self,
        f: Callable[[Array], float],
        grad: Callable[[Array], Array],
        hess: Callable[[Array], Array],
        options: Optional[CRNOptions] = None,
    ):
        """
        Parameters
        ----------
        f       : Callable[[array], float]  — objective function
        grad    : Callable[[array], array]  — gradient of f
        hess    : Callable[[array], array]  — Hessian of f (returns n×n array)
        options : CRNOptions, optional      — algorithm hyper-parameters;
                  defaults are used when omitted (sigma0=0.5, tol_grad=1e-6, max_iter=200)
        """
        self.f = f
        self.grad = grad
        self.hess = hess
        self.opt = options or CRNOptions()
        self.sigma = float(self.opt.sigma0)
        self.log: List[Dict[str, float]] = []
        self.runtime: Optional[float] = None
        self.termination_reason: Optional[str] = None

    def _model_value(self, f_x: float, g: Array, H: Array, h: Array, sigma: float) -> float:
        """Evaluate the cubic model m_k(h) = f + g^T h + 0.5 h^T H h + (sigma/3)||h||^3."""
        return float(f_x + g @ h + 0.5 * (h @ (H @ h)) + (sigma / 3.0) * (norm(h) ** 3))

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
        sigma = float(self.sigma)
        self.log = []
        start = time.perf_counter()
        reason = None

        for k in range(self.opt.max_iter):
            # (2) Evaluate model pieces at x_k.
            g = np.asarray(self.grad(x), dtype=float).reshape(-1)
            H = np.asarray(self.hess(x), dtype=float)
            H = 0.5 * (H + H.T)
            g_norm = float(norm(g))
            f_x = float(self.f(x))

            if g_norm <= self.opt.tol_grad:
                reason = "grad_tol"
                break

            # (3) Solve cubic subproblem for h_k (global minimizer).
            #     Bracket r by doubling and solve shifted systems with jitter for robustness.
            h = solve_cubic_subproblem(g, H, sigma)
            h_norm = float(norm(h))

            # (4) Majorization test and sigma update.
            f_trial = float(self.f(x + h))
            accepted = f_trial <= self._model_value(f_x, g, H, h, sigma)
            if accepted:
                x = x + h
                sigma = max(sigma / 2.0, self.opt.sigma_min)
            else:
                sigma = min(2.0 * sigma, self.opt.sigma_max)

            self.log.append(
                {
                    "iter": k,
                    "f": f_trial if accepted else f_x,
                    "g": g.copy(),
                    "H": H.copy(),
                    "grad_norm": g_norm,
                    "step_norm": h_norm,
                    "h": h.copy(),
                    "accepted": accepted,
                    "sigma": sigma,
                    "x": x.copy(),
                }
            )

            if h_norm <= self.opt.tol_step:
                reason = "step_tol"
                break

        self.sigma = sigma
        self.runtime = time.perf_counter() - start
        self.termination_reason = reason or "max_iter"
        return x
