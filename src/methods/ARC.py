"""
Adaptive Regularisation with Cubics (ARC)

Minimizes a smooth objective f by building a cubic model at each iterate,
computing a trial step, and deciding acceptance via the agreement ratio
rho = actual_reduction / predicted_reduction. Sigma is shrunk after very
successful steps (rho > eta2) and grown after unsuccessful ones (rho < eta1),
giving a self-tuning regularization schedule. Under Lipschitz-Hessian
assumptions this gives O(k^{-2}) convergence, matching NCR, but in practice
the adaptive sigma often converges faster on well-conditioned problems.

Reference: Cartis, Gould & Toint (2011). Math. Program. 127, 245–295.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.linalg import eigvalsh, norm, solve, LinAlgError

Array = np.ndarray
StepMethod = Literal["cauchy", "secular"]


def solve_cubic_subproblem(g: Array, H: Array, sigma: float) -> Array:
    """
    Global minimizer of m(h) = g^T h + 1/2 h^T H h + (sigma/3)||h||^3
    via the secular equation (Cartis-Gould-Toint style).
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    H = 0.5 * (np.asarray(H, dtype=float) + np.asarray(H, dtype=float).T)
    n = g.size
    if n == 0 or norm(g) == 0.0:
        return np.zeros_like(g)

    I = np.eye(n)

    lam_min = float(eigvalsh(H).min())
    r_low = max(0.0, (-lam_min) / max(sigma, 1e-300))

    def safe_solve(mat: Array, rhs: Array) -> Array:
        try:
            return solve(mat, rhs)
        except LinAlgError:
            jitter = 1e-12 * (1.0 + np.linalg.norm(mat, ord=np.inf))
            return solve(mat + jitter * I, rhs)

    def phi(r: float) -> float:
        s = safe_solve(H + (sigma * r) * I, g)
        return float(norm(s) - r)

    phi_low = phi(r_low)
    if phi_low <= 0.0:
        r_star = r_low
    else:
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

    h = -safe_solve(H + (sigma * r_star) * I, g)
    return h


@dataclass
class ARCParams:
    """
    Hyperparameters for AdaptiveCubicNewton (ARC).

    eta1, eta2       Step-acceptance thresholds: accept if rho >= eta1 (0 < eta1 <= eta2 < 1).
    gamma1, gamma2   Sigma scaling factors (1 < gamma1 <= gamma2): shrink by gamma1, grow by gamma2.
    sigma0           Initial regularization parameter (≈ 2·L3 if known; else 0.05).
    sigma_min        Floor for sigma to prevent it shrinking to zero.
    sigma_max        Cap for sigma to prevent unbounded growth (default: 5e11).
    tol_grad         Gradient norm stopping tolerance (||∇f|| <= tol_grad).
    tol_step         Step norm stopping tolerance (optional).
    tol_f            Function change stopping tolerance (optional, used together with tol_step).
    max_iter         Maximum number of iterations.
    verbose          Print per-iteration progress.
    bisect_tol       Bisection convergence tolerance for the secular equation.
    bisect_max_iter  Max bisection iterations for the secular equation.
    r_growth_max     Safety cap on the bracketing upper bound for the secular equation root.
    """
    eta1: float = 0.1
    eta2: float = 0.9
    gamma1: float = 2.0
    gamma2: float = 4.0
    sigma0: float = 0.05
    sigma_min: float = 1e-12
    sigma_max: float = 5e11
    tol_grad: float = 1e-8
    tol_step: Optional[float] = None
    tol_f: Optional[float] = None
    max_iter: int = 200
    verbose: bool = False
    bisect_tol: float = 1e-12
    bisect_max_iter: int = 200
    r_growth_max: float = 1e12


class AdaptiveCubicNewton:
    """
    Adaptive Regularisation with Cubics (ARC).

    Uses a ratio-based step acceptance test (rho = actual/predicted reduction)
    rather than the pure majorization test in NCR. This makes sigma adaptation
    more principled: the regularizer is tuned to the local model quality rather
    than only reacting to constraint violations. Supports both a cheap Cauchy
    step (gradient ray) and a globally optimal secular step as subproblem
    solvers.
    """

    # Algorithm structure (ARC):
    # (i) Step computation: build trial step h_k meeting at least Cauchy model decrease.
    # (ii) Model agreement: compute rho_k = (f(x_k)-f(x_k+h_k)) / (f(x_k)-m_k(h_k)).
    # (iii) Accept/reject: accept if rho_k >= eta1 (then x_{k+1}=x_k+h_k), else reject.
    # (iv) Sigma update: very successful -> shrink sigma; unsuccessful -> grow sigma; otherwise keep.

    def __init__(
        self,
        f: Callable[[Array], float],
        grad: Callable[[Array], Array],
        hess: Callable[[Array], Array],
        params: ARCParams = ARCParams(),
        step_method: StepMethod = "cauchy",
    ):
        """
        Parameters
        ----------
        f           : Callable[[array], float]   — objective function
        grad        : Callable[[array], array]   — gradient of f
        hess        : Callable[[array], array]   — Hessian of f (returns n×n array)
        params      : ARCParams, optional        — algorithm hyper-parameters;
                      defaults used when omitted (eta1=0.1, eta2=0.9, sigma0=0.05)
        step_method : "cauchy" | "secular"       — subproblem solver;
                      "cauchy" uses only gradient information (faster),
                      "secular" solves the full subproblem (more accurate)
        """
        self.f = f
        self.grad = grad
        self.hess = hess
        self.p = params
        self.step_method = step_method
        if not (0.0 < self.p.eta1 <= self.p.eta2 < 1.0):
            raise ValueError("Require 0 < eta1 <= eta2 < 1.")
        if not (1.0 < self.p.gamma1 <= self.p.gamma2):
            raise ValueError("Require 1 < gamma1 <= gamma2.")
        if self.p.sigma0 <= 0:
            raise ValueError("sigma0 must be positive.")
        self.log: List[Dict[str, object]] = []
        self.runtime: Optional[float] = None
        self.termination_reason: Optional[str] = None

    @staticmethod
    def _model_delta(g: Array, B: Array, s: Array, sigma: float) -> float:
        """Signed model change m_k(s) - m_k(0) = g^T s + 0.5 s^T B s + (sigma/3)||s||^3."""
        s_norm = float(norm(s))
        quad = 0.5 * float(s.T @ (B @ s))
        lin = float(g.T @ s)
        cub = (sigma / 3.0) * (s_norm**3)
        return lin + quad + cub

    @classmethod
    def _predicted_reduction(cls, g: Array, B: Array, s: Array, sigma: float) -> float:
        """Predicted reduction m_k(0) - m_k(s); used in the denominator of rho."""
        return -cls._model_delta(g, B, s, sigma)

    def _cauchy_step(self, g: Array, B: Array, sigma: float) -> Array:
        """Gradient-only Cauchy step along -g, scaled to minimize the cubic model on the gradient ray."""
        g_norm = float(norm(g))
        if g_norm == 0.0:
            return np.zeros_like(g)
        gBg = float(g.T @ (B @ g))
        a = sigma * (g_norm**3)
        b = gBg
        c = -(g_norm**2)
        if a <= 0.0:
            if b > 0:
                alpha = (g_norm**2) / b
            else:
                alpha = 1.0
            return -alpha * g
        disc = b * b - 4.0 * a * c
        disc = max(disc, 0.0)
        alpha_pos = (-b + np.sqrt(disc)) / (2.0 * a)
        alpha = max(alpha_pos, 0.0)
        return -alpha * g

    def _secular_step(self, g: Array, B: Array, sigma: float) -> Array:
        """Full Hessian-aware step via secular equation solver (globally optimal subproblem solution)."""
        if float(norm(g)) == 0.0:
            return np.zeros_like(g)
        # Delegate to module-level helper so tests can monkeypatch/log calls.
        return solve_cubic_subproblem(g, B, sigma)

    def _compute_trial_step(self, g: Array, B: Array, sigma: float) -> Array:
        """Dispatch to _cauchy_step or _secular_step based on self.step_method."""
        s_c = self._cauchy_step(g, B, sigma)
        if self.step_method == "cauchy":
            return s_c
        if self.step_method == "secular":
            s = self._secular_step(g, B, sigma)
            if self._model_delta(g, B, s, sigma) <= self._model_delta(g, B, s_c, sigma):
                return s
            return s_c
        raise ValueError(f"Unknown step_method: {self.step_method}")

    def _compute_rho(
        self,
        fx: float,
        fx_trial: float,
        g: Array,
        B: Array,
        s: Array,
        sigma: float,
    ) -> Tuple[float, float]:
        """Compute agreement ratio rho = actual_reduction / predicted_reduction."""
        actual = fx - fx_trial
        pred = self._predicted_reduction(g, B, s, sigma)
        if pred <= 0.0 or not np.isfinite(pred):
            return -np.inf, pred
        return float(actual / pred), float(pred)

    def _accept_step(self, rho: float) -> bool:
        """Return True if rho >= eta1 (step is at least mildly successful)."""
        return rho >= self.p.eta1

    def _update_sigma(self, sigma: float, rho: float) -> float:
        """Update sigma based on success category: very successful→shrink, unsuccessful→grow."""
        lo, hi = self.p.sigma_min, self.p.sigma_max
        if rho > self.p.eta2:
            # Very successful: shrink sigma.
            return max(lo, min(hi, sigma / self.p.gamma1))
        if rho >= self.p.eta1:
            # Successful: keep sigma.
            return max(lo, min(hi, sigma))
        if rho >= 0.0:
            # Unsuccessful but non-negative ratio: mild increase.
            return max(lo, min(hi, sigma * self.p.gamma1))
        # Negative ratio (step increased f): aggressive increase (gamma2).
        return max(lo, min(hi, sigma * self.p.gamma2))
    
    def _should_stop(
        self,
        g_norm: float,
        step_norm: float,
        x: Array,
        fx: float,
        fx_new: float,
    ) -> bool:
        """Check all termination criteria; return True if any are satisfied."""
        if g_norm <= self.p.tol_grad:
            return True
        if self.p.tol_step is not None and self.p.tol_f is not None:
            if step_norm <= self.p.tol_step * (1.0 + float(norm(x))):
                if abs(fx_new - fx) <= self.p.tol_f * (1.0 + abs(fx)):
                    return True
        if self.p.tol_step is not None and self.p.tol_f is None:
            if step_norm <= self.p.tol_step:
                return True
        return False

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
        sigma = float(self.p.sigma0)
        self.log = []
        start_time = time.perf_counter()
        reason = None

        for k in range(self.p.max_iter):
            # (i) Evaluate model pieces at x_k.
            fx = float(self.f(x))
            g = np.asarray(self.grad(x), dtype=float).reshape(-1)
            # store self.hess for computational purposes so B is symmetric.
            _H = np.asarray(self.hess(x), dtype=float)
            B = 0.5 * (_H + _H.T)
            g_norm = float(norm(g))
            if g_norm <= self.p.tol_grad:
                reason = "grad_tol"
                break

            # (i) Compute trial step meeting Cauchy decrease.
            s = self._compute_trial_step(g, B, sigma)
            x_trial = x + s
            fx_trial = float(self.f(x_trial))
            # (ii) Model agreement.
            rho, pred = self._compute_rho(fx, fx_trial, g, B, s, sigma)
            # (iii) Accept / reject.
            accepted = self._accept_step(rho)
            x_next = x_trial if accepted else x
            # (iv) Update sigma based on success category.
            sigma_next = self._update_sigma(sigma, rho)
            step_norm = float(norm(s))
            stop_now = self._should_stop(
                g_norm=g_norm, step_norm=step_norm, x=x, fx=fx, fx_new=fx_trial
            )

            self.log.append(
                dict(
                    iter=k,
                    f=fx,
                    g=g.copy(),
                    H=B.copy(),
                    grad_norm=g_norm,
                    step_norm=step_norm,
                    h=s.copy(),
                    x_trial=x_trial.copy(),
                    accepted=accepted,
                    sigma=sigma,
                    rho=rho,
                    predicted_reduction=pred,
                    x=x.copy(),
                )
            )

            if self.p.verbose:
                print(
                    f"[k={k:03d}] f={fx:.6e} ||g||={g_norm:.3e} "
                    f"sigma={sigma:.3e} ||s||={step_norm:.3e} rho={rho:.3e} "
                    f"{'ACCEPT' if accepted else 'REJECT'}"
                )

            x, sigma = x_next, sigma_next
            if stop_now:
                reason = reason or "stopping_criteria"
                break

        self.runtime = time.perf_counter() - start_time
        self.termination_reason = reason or "max_iter"
        return x

