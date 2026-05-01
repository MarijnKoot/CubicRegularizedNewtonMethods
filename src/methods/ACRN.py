"""
Accelerated Cubic-Regularized Newton (ACRN)

Implements Nesterov's estimate-sequence acceleration applied to the
cubic-regularized Newton step. At each outer iteration k, the algorithm
maintains two sequences:
  y_k  — the inner cubic-step point (where the model is built and minimized)
  v_k  — a minimizer estimate updated via a gradient accumulation scheme

The mixing coefficient alpha_k interpolates between x_k and v_k to form y_k,
which is then passed to the cubic subproblem solver to produce x_{k+1}.
This gives O(k^{-3}) convergence in function value on convex problems, an
improvement over the O(k^{-2}) of NCR and ARC.

Requires convexity and a known or estimated Hessian Lipschitz constant L3.
Benchmark scripts verify min_eig(H(x0)) >= -1e-6 before running ACRN.

Reference: Nesterov (2008). Math. Program. 112, 159–181.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError, eigvalsh, norm, solve

Array = np.ndarray


def _as_1d_array(x: Array | float | List[float]) -> Array:
    arr = np.asarray(x, dtype=float)
    return np.ascontiguousarray(arr.reshape(-1))


def _symmetrize(H: Array) -> Array:
    return 0.5 * (H + H.T)


def solve_cubic_subproblem(g: Array, H: Array, sigma: float) -> Array:
    """Minimal wrapper used in tests; solves full cubic model."""
    solver = CubicSubproblemSolver(sigma=sigma)
    return solver.solve(g=_as_1d_array(g), H=np.asarray(H, dtype=float)).h


@dataclass
class CubicSubproblemResult:
    """Named result of the cubic subproblem solver: h (step vector) and r (step norm ||h||)."""
    h: Array
    r: float


class CubicSubproblemSolver:
    """Thin object-oriented wrapper around the secular-equation cubic subproblem for use inside ACRN."""

    def __init__(
        self,
        sigma: float,
        tol_phi: float = 1e-12,
        max_bisect_iter: int = 200,
        bracket_growth: float = 2.0,
        r_eps: float = 1e-12,
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.sigma = float(sigma)
        self.tol_phi = float(tol_phi)
        self.max_bisect_iter = int(max_bisect_iter)
        self.bracket_growth = float(bracket_growth)
        self.r_eps = float(r_eps)

    def solve(self, g: Array, H: Array) -> CubicSubproblemResult:
        g = _as_1d_array(g)
        n = g.size
        if n == 0 or norm(g) == 0.0:
            return CubicSubproblemResult(h=np.zeros_like(g), r=0.0)

        H = _symmetrize(np.asarray(H, dtype=float))
        if H.shape != (n, n):
            raise ValueError(f"H has shape {H.shape}, expected {(n, n)}.")

        lam_min = float(np.min(eigvalsh(H)))
        r_low = max(0.0, (-lam_min / self.sigma)) + self.r_eps
        I = np.eye(n)

        def safe_solve(A: Array, b: Array) -> Array:
            try:
                return solve(A, b)
            except LinAlgError:
                jitter = 1e-12 * (1.0 + np.linalg.norm(A, ord=np.inf))
                return solve(A + jitter * I, b)

        def phi(r: float) -> float:
            A = H + (self.sigma * r) * I
            u = safe_solve(A, g)
            return float(norm(-u) - r)

        phi_low = phi(r_low)
        if phi_low <= 0.0:
            A = H + (self.sigma * r_low) * I
            h = -safe_solve(A, g)
            return CubicSubproblemResult(h=h, r=float(r_low))

        r_high = max(1.0, r_low * 2.0)
        phi_high = phi(r_high)
        while phi_high > 0.0:
            r_high *= self.bracket_growth
            phi_high = phi(r_high)
            if r_high > 1e20:
                raise RuntimeError("Failed to bracket secular equation root.")

        low, high = r_low, r_high
        mid = 0.5 * (low + high)
        for _ in range(self.max_bisect_iter):
            mid = 0.5 * (low + high)
            val = phi(mid)
            if abs(val) <= self.tol_phi:
                break
            if val > 0.0:
                low = mid
            else:
                high = mid
            if (high - low) <= max(self.r_eps, 10.0 * self.tol_phi * (1.0 + mid)):
                break

        r_star = float(mid)
        A = H + (self.sigma * r_star) * I
        h_star = -safe_solve(A, g)
        return CubicSubproblemResult(h=h_star, r=r_star)


@dataclass
class ACRNHistoryRow:
    """Per-outer-iteration trace row for ACRN (one row per k, stored in the info dict returned by run)."""
    k: int
    xk: Array
    yk: Array
    hk: Array
    f_yk: float
    grad_yk_norm: float
    hess_min_eig_yk: float
    sigma_k: float
    sigma_next: float
    rejected_trials: int


class AcceleratedCubicNewton:
    """
    Explicit accelerated cubic-regularized Newton (linear-estimate variant).

    Implements Nesterov (2008)'s estimate-sequence scheme: a two-sequence
    structure (x_k = inner iterate, v_k = minimizer estimate) with a mixing
    step y_k = (1-alpha_k)*x_k + alpha_k*v_k before each cubic solve.
    Requires convexity and a known or numerically estimated Hessian Lipschitz
    constant L3. Unlike NCR and ARC, the step is always accepted (no rejection
    test in the fixed-sigma mode).
    """

    def __init__(
        self,
        f: Callable[[Array], float],
        grad: Callable[[Array], Array],
        hess: Callable[[Array], Array],
        L3: float,
        *,
        sigma: Optional[float] = None,
        N: Optional[float] = None,
        tol_grad: float = 1e-8,
        tol_step: float = 1e-12,
        max_iter: int = 200,
        verbose: bool = False,
        adaptive_sigma: bool = False,
        sigma_min: float = 1e-15,
        sigma_max: float = 5e11,
    ) -> None:
        """
        Parameters
        ----------
        f              : Callable[[array], float]  — objective function
        grad           : Callable[[array], array]  — gradient of f
        hess           : Callable[[array], array]  — Hessian of f (returns n×n array)
        L3             : float                     — Lipschitz constant of the Hessian;
                         must be positive. Use utilities.estimate_L3 if unknown.
        sigma          : float, optional           — initial regularization parameter;
                         defaults to L3 (theory-optimal for fixed sigma)
        N              : float, optional           — estimate-sequence curvature parameter;
                         defaults to 12*L3
        tol_grad       : float                     — gradient-norm stopping tolerance (default 1e-8)
        tol_step       : float                     — relative step-norm stopping tolerance (default 1e-12)
        max_iter       : int                       — maximum outer iterations (default 200)
        verbose        : bool                      — print per-iteration diagnostics (default False)
        adaptive_sigma : bool                      — enable CRN-style sigma adaptation (default False);
                         when True, sigma is halved on accepted steps and doubled on rejected ones
        sigma_min      : float                     — floor for sigma in adaptive mode (default 1e-15)
        sigma_max      : float                     — cap for sigma in adaptive mode (default 5e11)
        """
        self.f = f
        self.grad = grad
        self.hess = hess
        self.L3 = float(L3)
        self.sigma = float(self.L3 if sigma is None else sigma)
        self.N = float(12.0 * self.L3 if N is None else N)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        if self.sigma <= 0 or self.N <= 0:
            raise ValueError("sigma and N must be positive.")
        if self.sigma_min <= 0 or self.sigma_max < self.sigma_min:
            raise ValueError("Require 0 < sigma_min <= sigma_max.")
        self.tol_grad = float(tol_grad)
        self.tol_step = float(tol_step)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.adaptive_sigma = bool(adaptive_sigma)
        self._N_over_sigma = self.N / self.sigma
        self.log: List[Dict[str, object]] = []
        self.termination_reason: Optional[str] = None

    @staticmethod
    def _a_k(k: int) -> float:
        """Weight a_k = (k+1)(k+2)/2 in the estimate-sequence sum A_k = sum_{i=0}^{k} a_i."""
        return ((k + 1) * (k + 2)) / 2.0

    @staticmethod
    def _alpha_k(k: int) -> float:
        """Mixing coefficient alpha_k = 3/(k+3) used to blend x_k and v_k into y_k."""
        return 3.0 / (k + 3.0)

    @staticmethod
    def _model_value(f_y: float, g: Array, H: Array, h: Array, sigma: float) -> float:
        """Evaluate the cubic model m(h) = f_y + g^T h + 0.5 h^T H h + (sigma/3)||h||^3 at step h from y."""
        return float(f_y + g @ h + 0.5 * (h @ (H @ h)) + (sigma / 3.0) * norm(h) ** 3)

    def _sync_N_from_sigma(self) -> None:
        """Keep N proportional to sigma (N = (N/sigma_init)*sigma) when adaptive_sigma=True."""
        self.N = self._N_over_sigma * self.sigma

    def _cubic_step_fixed(self, y: Array) -> Tuple[Array, Dict[str, object]]:
        """Compute cubic step with fixed sigma (no adaptation); always accepted."""
        g = _as_1d_array(self.grad(y))
        H = _symmetrize(np.asarray(self.hess(y), dtype=float))
        sigma_k = float(self.sigma)
        h = solve_cubic_subproblem(g, H, sigma_k)
        f_y = float(self.f(y))
        f_trial = float(self.f(y + h))
        return y + h, {
            "g": g,
            "H": H,
            "h": h,
            "f_y": f_y,
            "f_trial": f_trial,
            "sigma_k": sigma_k,
            "sigma_next": sigma_k,
            "rejected_trials": 0,
        }

    def _cubic_step_adaptive(self, y: Array) -> Tuple[Array, Dict[str, object]]:
        """CRN-style sigma adaptation at fixed y: double-on-fail, halve-on-accept."""
        g = _as_1d_array(self.grad(y))
        H = _symmetrize(np.asarray(self.hess(y), dtype=float))
        f_y = float(self.f(y))
        rejected_trials = 0

        if not np.isfinite(f_y) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(H)):
            raise RuntimeError("ACRN: overflow in f/grad/hess at y — function not finite at this iterate.")

        while True:
            sigma_k = float(self.sigma)
            h = solve_cubic_subproblem(g, H, sigma_k)
            f_trial = float(self.f(y + h))
            accepted = f_trial <= self._model_value(f_y, g, H, h, sigma_k)
            if accepted:
                self.sigma = max(sigma_k / 2.0, self.sigma_min)
                self._sync_N_from_sigma()
                return y + h, {
                    "g": g,
                    "H": H,
                    "h": h,
                    "f_y": f_y,
                    "f_trial": f_trial,
                    "sigma_k": sigma_k,
                    "sigma_next": float(self.sigma),
                    "rejected_trials": rejected_trials,
                }

            if sigma_k >= self.sigma_max:
                raise RuntimeError("ACRN failed to satisfy model majorization before sigma_max.")

            self.sigma = min(2.0 * sigma_k, self.sigma_max)
            self._sync_N_from_sigma()
            rejected_trials += 1

    def _estimate_minimizer(self, p: Array, x0: Array) -> Array:
        """
        Advance the minimizer estimate using the accumulated gradient sum p.

        Computes v = x0 - sqrt(2*||p||/N) * (p/||p||), the minimizer of the
        linear estimate-sequence function centered at x0 with curvature N.
        Falls back to x0 when the update would be non-finite or unreasonably large.
        """
        p = _as_1d_array(p)
        p_norm = float(norm(p))
        if p_norm == 0.0 or not np.isfinite(p_norm):
            return x0.copy()
        direction = p / p_norm
        if not np.all(np.isfinite(direction)):
            return x0.copy()
        scale_sq = 2.0 * p_norm / self.N
        if not np.isfinite(scale_sq):
            return x0.copy()
        scale = np.sqrt(scale_sq)
        max_scale = 1e8 * (1.0 + float(norm(x0)))
        if not np.isfinite(scale) or scale > max_scale:
            return x0.copy()
        candidate = x0 - scale * direction
        if not np.all(np.isfinite(candidate)):
            return x0.copy()
        return candidate

    def run(self, x0: Array) -> Tuple[Array, Dict[str, object]]:
        """
        Run the solver from initial point x0.

        Parameters
        ----------
        x0 : array-like, shape (n,)
            Starting point. Should be in the convex domain of f.

        Returns
        -------
        x    : ndarray, shape (n,)
            Approximate minimizer at termination.
        info : dict
            Summary dict with keys: 'iterations', 'history' (list[ACRNHistoryRow]),
            'final_grad_norm', 'sigma', 'N', 'sigma_history', 'note'.

        Side effects
        ------------
        Populates self.log and self.termination_reason.
        (runtime is not stored; wrap in time.perf_counter() externally if needed.)
        """
        x0 = _as_1d_array(x0)
        history: List[ACRNHistoryRow] = []
        self.log = []
        self.termination_reason = None

        step_fn = self._cubic_step_adaptive if self.adaptive_sigma else self._cubic_step_fixed

        x1, init_info = step_fn(x0)
        p = np.zeros_like(x0)
        v = x0.copy()

        g0 = _as_1d_array(init_info["g"])
        H0 = _symmetrize(np.asarray(init_info["H"], dtype=float))
        h0 = _as_1d_array(init_info["h"])
        sigma_k0 = float(init_info["sigma_k"])
        sigma_next0 = float(init_info["sigma_next"])
        rejected0 = int(init_info["rejected_trials"])
        f0 = float(init_info["f_trial"])

        history.append(
            ACRNHistoryRow(
                k=0,
                xk=x0.copy(),
                yk=x0.copy(),
                hk=h0.copy(),
                f_yk=f0,
                grad_yk_norm=float(norm(g0)),
                hess_min_eig_yk=float(np.min(eigvalsh(H0))) if g0.size else 0.0,
                sigma_k=sigma_k0,
                sigma_next=sigma_next0,
                rejected_trials=rejected0,
            )
        )
        self.log.append(
            {
                "iter": 0,
                "f": f0,
                "g": g0.copy(),
                "H": H0.copy(),
                "grad_norm": float(norm(g0)),
                "step_norm": float(norm(h0)),
                "h": h0.copy(),
                "accepted": True,
                "sigma_trial": sigma_k0,
                "sigma": sigma_next0,
                "rejected_trials": rejected0,
                "x": x1.copy(),
            }
        )

        x = x1.copy()
        g_at_x = _as_1d_array(self.grad(x))
        if norm(g_at_x) <= self.tol_grad:
            self.termination_reason = "grad_tol"
            return x, {
                "iterations": 1,
                "history": history,
                "final_grad_norm": float(norm(g_at_x)),
                "sigma": self.sigma,
                "N": self.N,
                "sigma_history": [entry["sigma"] for entry in self.log],
                "note": "Stopped after initialization step.",
            }

        # Outer acceleration loop.
        # Two-sequence structure (Nesterov 2008):
        #   x_k  — the current best iterate (cubic step output)
        #   v_k  — the minimizer estimate (updated via gradient accumulation p)
        # At each step: y_k = (1-alpha_k)*x_k + alpha_k*v_k  (mixing step)
        #               x_{k+1} = y_k + h_k  (cubic step from y_k)
        #               p += a_k * grad(x_{k+1})
        #               v_{k+1} = estimate_minimizer(p, x0)
        for k in range(1, self.max_iter + 1):
            g_at_x = _as_1d_array(self.grad(x))
            gnorm = float(norm(g_at_x))
            if gnorm <= self.tol_grad:
                self.termination_reason = "grad_tol"
                if self.verbose:
                    print(f"[stop] k={k}: ||grad(xk)||={gnorm:.3e}")
                break

            alpha = self._alpha_k(k)
            y = (1.0 - alpha) * x + alpha * v
            if not np.all(np.isfinite(y)):
                y = x.copy()

            try:
                x_next, step_info = step_fn(y)
            except RuntimeError:
                self.termination_reason = "non_finite"
                break
            gy = _as_1d_array(step_info["g"])
            Hy = _symmetrize(np.asarray(step_info["H"], dtype=float))
            h = _as_1d_array(step_info["h"])
            f_y = float(step_info["f_y"])
            f_next = float(step_info["f_trial"])
            sigma_k = float(step_info["sigma_k"])
            sigma_next = float(step_info["sigma_next"])
            rejected_trials = int(step_info["rejected_trials"])

            history.append(
                ACRNHistoryRow(
                    k=k,
                    xk=x.copy(),
                    yk=y.copy(),
                    hk=h.copy(),
                    f_yk=f_y,
                    grad_yk_norm=float(norm(gy)),
                    hess_min_eig_yk=float(np.min(eigvalsh(Hy))) if gy.size else 0.0,
                    sigma_k=sigma_k,
                    sigma_next=sigma_next,
                    rejected_trials=rejected_trials,
                )
            )
            self.log.append(
                {
                    "iter": k,
                    "f": f_next,
                    "g": gy.copy(),
                    "H": Hy.copy(),
                    "grad_norm": float(norm(gy)),
                    "step_norm": float(norm(h)),
                    "h": h.copy(),
                    "accepted": True,
                    "sigma_trial": sigma_k,
                    "sigma": sigma_next,
                    "rejected_trials": rejected_trials,
                    "x": x_next.copy(),
                }
            )

            if norm(x_next - x) <= self.tol_step * (1.0 + norm(x)):
                x = x_next
                self.termination_reason = "step_tol"
                if self.verbose:
                    print(f"[stop] k={k}: tiny step.")
                break

            a = self._a_k(k)
            p = p + a * _as_1d_array(self.grad(x_next))
            v = self._estimate_minimizer(p, x0=x0)

            x = x_next
            if self.verbose and (k <= 5 or k % 10 == 0):
                print(
                    f"k={k:3d}  alpha={alpha:.4f}  "
                    f"||grad(x)||={float(norm(self.grad(x))):.3e}  x={x}"
                )

        if self.termination_reason is None:
            self.termination_reason = "max_iter"
        return x, {
            "iterations": max(0, len(history) - 1),
            "history": history,
            "final_grad_norm": float(norm(_as_1d_array(self.grad(x)))),
            "sigma": self.sigma,
            "N": self.N,
            "sigma_history": [entry["sigma"] for entry in self.log],
            "note": self.termination_reason,
        }
