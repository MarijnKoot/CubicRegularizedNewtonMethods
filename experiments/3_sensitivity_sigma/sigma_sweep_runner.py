"""
sigma_sweep_runner.py — Core logic for the σ₀ grid-sweep experiment.

For each (problem, method, σ₀, seed) combination, runs the solver once and
returns a structured SigmaResult.  The σ₀ grid is log-spaced, centred around the
median σ_ref = 2·L3 across the problem set (~4e-4 … ~4e7).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from numpy.linalg import norm

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from harness.problems  import get_problem
from harness.counters  import EvalCounter
from utilities         import estimate_L3
from methods.pure_newton import PureNewton, PureNewtonOptions
from methods.NCR         import CubicNewton, CRNOptions
from methods.ARC         import AdaptiveCubicNewton, ARCParams
from methods.ACRN        import AcceleratedCubicNewton

# ── constants ─────────────────────────────────────────────────────────────────
EPS_G    = 1e-7
MAX_ITER = 500
N_SEEDS  = 5
SEEDS    = list(range(42, 42 + N_SEEDS))

# σ₀ grid: 12 log-spaced values centred around the median σ_ref across problems.
# σ_ref values: ~1 (Rastrigin), ~10 (LogSumExp), ~5000 (Rosenbrock) → log10 centre ≈ 2.1.
# Grid spans ±5.5 decades around that centre: ~4e-4 … ~4e7.
SIGMA_GRID = np.logspace(-3.4, 7.6, 12).tolist()

METHODS = ["Newton", "NCR", "ARC", "ACRN"]   # Newton is σ-independent (reference)

# Problems: (registry_name, n, problem_kwargs, label)
PROBLEMS = [
    ("quadratic",  10, {"cond": 1e2},  "Quadratic κ=1e2"),
    ("quadratic",  10, {"cond": 1e6},  "Quadratic κ=1e6"),
    ("logsumexp",  10, {},             "LogSumExp n=10"),
    ("rosenbrock",  2, {},             "Rosenbrock n=2"),
    ("rosenbrock", 10, {},             "Rosenbrock n=10"),
    ("rastrigin",  10, {},             "Rastrigin n=10"),
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SigmaResult:
    problem:       str
    label:         str
    n:             int
    method:        str
    sigma0:        float   # the σ₀ value used (nan for Newton)
    sigma_ref:     float   # 2·L3 reference (nan if unknown)
    seed:          int
    success:       bool
    termination_reason: str
    iterations:    int
    rejected_iter: int
    runtime_sec:   float
    final_f:       float
    final_grad_norm: float
    numerical_failure: bool

    def to_row(self) -> dict:
        return {
            "problem":            self.problem,
            "label":              self.label,
            "n":                  self.n,
            "method":             self.method,
            "sigma0":             self.sigma0,
            "sigma_ref":          self.sigma_ref,
            "seed":               self.seed,
            "success":            int(self.success),
            "termination_reason": self.termination_reason,
            "iterations":         self.iterations,
            "rejected_iter":      self.rejected_iter,
            "runtime_sec":        round(self.runtime_sec, 6),
            "final_f":            self.final_f,
            "final_grad_norm":    self.final_grad_norm,
            "numerical_failure":  int(self.numerical_failure),
        }

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "problem", "label", "n", "method", "sigma0", "sigma_ref", "seed",
            "success", "termination_reason",
            "iterations", "rejected_iter", "runtime_sec",
            "final_f", "final_grad_norm", "numerical_failure",
        ]


# ── starting point per problem ─────────────────────────────────────────────────

def _x0_for(problem: str, n: int, seed: int) -> np.ndarray:
    if problem == "rosenbrock":
        return np.full(n, -1.0)
    rng = np.random.default_rng(seed + 1000)
    return rng.standard_normal(n)


# ── single run ────────────────────────────────────────────────────────────────

def run_one(
    problem_name: str,
    label: str,
    n: int,
    problem_kwargs: dict,
    method: str,
    sigma0: float,
    seed: int,
) -> SigmaResult:

    f_raw, grad_raw, hess_raw, meta = get_problem(
        problem_name, n, seed=seed, problem_kwargs=problem_kwargs
    )
    x0 = _x0_for(problem_name, n, seed)
    counter = EvalCounter(f_raw, grad_raw, hess_raw)

    # Estimate σ_ref = 2·L3 for reference marking on plots
    L3_known = meta.get("L3")
    if L3_known is not None and L3_known > 0:
        sigma_ref = 2.0 * float(L3_known)
    elif L3_known == 0.0:
        sigma_ref = 0.0   # quadratic: L3=0 exactly
    else:
        try:
            sigma_ref = 2.0 * float(estimate_L3(hess_raw, x0))
        except Exception:
            sigma_ref = float("nan")

    t0 = time.perf_counter()
    try:
        if method == "Newton":
            opts   = PureNewtonOptions(tol_grad=EPS_G, max_iter=MAX_ITER)
            solver = PureNewton(counter.f, counter.grad, counter.hess, options=opts)
            x_final = solver.run(x0.copy())
            iters   = len(solver.log)
            rejected = 0
            reason  = solver.termination_reason or "max_iter"

        elif method == "NCR":
            s = max(float(sigma0), 1e-15)
            opts = CRNOptions(
                sigma0=s, sigma_min=1e-15, sigma_max=5e11,
                tol_grad=EPS_G, max_iter=MAX_ITER,
            )
            solver  = CubicNewton(counter.f, counter.grad, counter.hess, options=opts)
            x_final = solver.run(x0.copy())
            iters   = len(solver.log)
            rejected = sum(1 for e in solver.log if not e.get("accepted", True))
            reason  = solver.termination_reason or "max_iter"

        elif method == "ARC":
            s = max(float(sigma0), 1e-15)
            params  = ARCParams(
                sigma0=s, sigma_min=1e-15, sigma_max=5e11,
                eta1=0.1, eta2=0.9, tol_grad=EPS_G, max_iter=MAX_ITER,
            )
            solver  = AdaptiveCubicNewton(
                counter.f, counter.grad, counter.hess,
                params=params, step_method="secular",
            )
            x_final = solver.run(x0.copy())
            iters   = len(solver.log)
            rejected = sum(1 for e in solver.log if not e.get("accepted", True))
            reason  = solver.termination_reason or "max_iter"

        elif method == "ACRN":
            # Run on all problems; non-convex ones will simply fail to converge.
            L3_eff = max(sigma_ref / 2.0, 1e-15)   # sigma_ref = 2*L3 → L3 = sigma_ref/2
            s = max(float(sigma0), 1e-15)
            solver = AcceleratedCubicNewton(
                counter.f, counter.grad, counter.hess,
                L3=L3_eff, sigma=s,
                sigma_min=1e-15, sigma_max=5e11,
                tol_grad=EPS_G, max_iter=MAX_ITER,
                adaptive_sigma=True,
            )
            x_final, _ = solver.run(x0.copy())
            iters    = len(solver.log)
            rejected = sum(int(e.get("rejected_trials", 0)) for e in solver.log)
            reason   = solver.termination_reason or "max_iter"

        else:
            raise ValueError(f"Unknown method: {method!r}")

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return SigmaResult(
            problem=problem_name, label=label, n=n,
            method=method, sigma0=sigma0, sigma_ref=sigma_ref, seed=seed,
            success=False, termination_reason=f"error:{str(exc)[:80]}",
            iterations=0, rejected_iter=0, runtime_sec=elapsed,
            final_f=float("nan"), final_grad_norm=float("nan"),
            numerical_failure=True,
        )

    elapsed = time.perf_counter() - t0
    num_fail = not np.all(np.isfinite(x_final))
    final_f  = float(f_raw(x_final)) if not num_fail else float("nan")
    final_gn = float(norm(grad_raw(x_final))) if not num_fail else float("nan")
    success  = np.isfinite(final_gn) and final_gn <= EPS_G

    return SigmaResult(
        problem=problem_name, label=label, n=n,
        method=method, sigma0=sigma0, sigma_ref=sigma_ref, seed=seed,
        success=success, termination_reason=reason,
        iterations=iters, rejected_iter=rejected, runtime_sec=elapsed,
        final_f=final_f, final_grad_norm=final_gn,
        numerical_failure=num_fail,
    )


# ── full sweep ────────────────────────────────────────────────────────────────

def run_all(verbose: bool = True) -> List[SigmaResult]:
    results = []
    for prob_name, n, prob_kwargs, label in PROBLEMS:
        if verbose:
            print(f"\n--- {label} ---")
        for method in METHODS:
            sigma_iter = [float("nan")] if method == "Newton" else SIGMA_GRID
            for sigma0 in sigma_iter:
                for seed in SEEDS:
                    r = run_one(prob_name, label, n, prob_kwargs,
                                method, sigma0, seed)
                    results.append(r)
            if verbose:
                ok = sum(1 for r in results
                         if r.label == label and r.method == method and r.success)
                tot = len(results) - sum(
                    1 for r in results if r.label != label or r.method != method
                )
                # simpler count
                n_runs = sum(1 for r in results if r.label == label and r.method == method)
                n_ok   = sum(1 for r in results if r.label == label and r.method == method and r.success)
                print(f"  {method:8s}: {n_ok}/{n_runs} converged")
    return results
