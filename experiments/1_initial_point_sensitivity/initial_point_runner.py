"""
initial_point_runner.py — core logic for the initial point sensitivity experiment.

For each (problem, method, starting point) triple, runs the solver once and
returns a structured result dict. Starting points are:
  0 : standard benchmark start (problem-specific)
  1 : benign start (closer to the known/estimated minimizer)
  2–N : random Gaussian perturbations of the standard start

All randomness is controlled by SEED for reproducibility.
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
from harness.adapters  import run_and_extract
from utilities         import estimate_L3

# ── constants ─────────────────────────────────────────────────────────────────
SEED      = 42
N_RANDOM  = 5       # number of random perturbation starts
PERTURB   = 2.0     # std of Gaussian perturbation added to standard start
EPS_G     = 1e-7    # gradient-norm stopping tolerance (grad-only, matches stopping criteria experiment)
MAX_ITER  = 500
DIM       = 10      # problem dimension

METHODS = ["Newton", "NCR", "ARC", "ACRN"]

# Problems: (registry_name, label, problem_kwargs)
PROBLEMS = [
    ("logsumexp",      "LogSumExp",      {}),
    ("quartic_convex", "QuarticConvex",  {}),
    ("rosenbrock",     "Rosenbrock",     {}),
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class InitPointResult:
    problem:       str
    method:        str
    x0_index:      int     # 0=standard, 1=benign, 2..N=random
    x0_label:      str     # "standard" | "benign" | "random_0" etc.
    x0_norm:       float   # ||x0||
    success:       bool
    termination_reason: str
    iterations:    int
    runtime_sec:   float
    final_f:       float
    final_grad_norm: float
    rejected_iter: int

    def to_row(self) -> dict:
        return {
            "problem":            self.problem,
            "method":             self.method,
            "x0_index":           self.x0_index,
            "x0_label":           self.x0_label,
            "x0_norm":            round(self.x0_norm, 6),
            "success":            int(self.success),
            "termination_reason": self.termination_reason,
            "iterations":         self.iterations,
            "runtime_sec":        self.runtime_sec,
            "final_f":            self.final_f,
            "final_grad_norm":    self.final_grad_norm,
            "rejected_iter":      self.rejected_iter,
        }

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "problem", "method", "x0_index", "x0_label", "x0_norm",
            "success", "termination_reason",
            "iterations", "runtime_sec",
            "final_f", "final_grad_norm", "rejected_iter",
        ]


# ── starting point generation ─────────────────────────────────────────────────

def make_starting_points(problem: str, n: int) -> List[tuple[str, np.ndarray]]:
    """
    Returns a list of (label, x0) pairs for a given problem.

    Index 0 : standard start  — problem-specific, common in literature
    Index 1 : benign start    — closer to the minimizer
    Index 2+ : random starts  — standard start + N(0, PERTURB^2) noise
    """
    rng = np.random.default_rng(SEED)

    if problem == "rosenbrock":
        x_star   = np.ones(n)                 # global min at (1,...,1)
        standard = np.full(n, -1.0)           # classic hard start, dist=2.83
        benign   = x_star + rng.standard_normal(n) * 0.2   # near x*
    elif problem == "quartic_convex":
        x_star   = np.zeros(n)               # global min at origin
        standard = np.ones(n) * 3.0          # away from origin, dist=4.24
        benign   = x_star + rng.standard_normal(n) * 0.2   # near x*
    elif problem == "logsumexp":
        # estimate x* with L-BFGS from a neutral start
        from scipy.optimize import minimize as _minimize
        f_raw, grad_raw, _, _ = get_problem(problem, n, seed=SEED)
        res    = _minimize(f_raw, np.zeros(n), jac=grad_raw, method="L-BFGS-B")
        x_star = res.x
        standard = np.zeros(n)               # common neutral start, dist~0.84
        benign   = x_star + rng.standard_normal(n) * 0.2   # near x*
    else:
        x_star   = np.zeros(n)
        standard = np.ones(n)
        benign   = x_star + rng.standard_normal(n) * 0.2

    starts = [("standard", standard), ("benign", benign)]
    for i in range(N_RANDOM):
        noise = rng.standard_normal(n) * PERTURB
        starts.append((f"random_{i}", standard + noise))

    return starts


# ── single run ────────────────────────────────────────────────────────────────

def run_one(
    problem_name: str,
    method:       str,
    x0_label:     str,
    x0_index:     int,
    x0:           np.ndarray,
    verbose:      bool = False,
) -> InitPointResult:
    f_raw, grad_raw, hess_raw, meta = get_problem(
        problem_name, len(x0), seed=SEED
    )
    counter = EvalCounter(f_raw, grad_raw, hess_raw)
    L3 = meta.get("L3")

    # ACRN only valid for convex problems
    if method == "ACRN" and not meta.get("convex", False):
        return InitPointResult(
            problem=problem_name, method=method,
            x0_index=x0_index, x0_label=x0_label,
            x0_norm=float(norm(x0)),
            success=False,
            termination_reason="error:ACRN requires convex problem",
            iterations=0, runtime_sec=float("nan"),
            final_f=float("nan"), final_grad_norm=float("nan"),
            rejected_iter=0,
        )

    try:
        raw = run_and_extract(
            method=method,
            counter=counter,
            x0=x0.copy(),
            eps_g=EPS_G,
            max_iter=MAX_ITER,
            L3=L3,
            solver_kwargs={},
        )
    except Exception as exc:
        return InitPointResult(
            problem=problem_name, method=method,
            x0_index=x0_index, x0_label=x0_label,
            x0_norm=float(norm(x0)),
            success=False,
            termination_reason=f"error:{str(exc)[:80]}",
            iterations=0, runtime_sec=float("nan"),
            final_f=float("nan"), final_grad_norm=float("nan"),
            rejected_iter=0,
        )

    x_final      = raw["x_final"]
    final_f      = float(f_raw(x_final))
    final_gnorm  = float(norm(grad_raw(x_final)))
    success      = final_gnorm <= EPS_G

    return InitPointResult(
        problem=problem_name, method=method,
        x0_index=x0_index, x0_label=x0_label,
        x0_norm=float(norm(x0)),
        success=success,
        termination_reason=raw["stop_reason"],
        iterations=raw["outer_iterations"],
        runtime_sec=raw["elapsed"],
        final_f=final_f,
        final_grad_norm=final_gnorm,
        rejected_iter=raw["rejected_iterations"],
    )


# ── full experiment ───────────────────────────────────────────────────────────

def run_all(verbose: bool = True) -> List[InitPointResult]:
    results = []
    for prob_name, prob_label, _ in PROBLEMS:
        starts = make_starting_points(prob_name, DIM)
        if verbose:
            print(f"\n--- {prob_label} (n={DIM}) ---")
        for method in METHODS:
            for idx, (label, x0) in enumerate(starts):
                r = run_one(prob_name, method, label, idx, x0, verbose=False)
                results.append(r)
            if verbose:
                n_ok = sum(1 for r in results
                           if r.problem == prob_name and r.method == method
                           and r.success)
                n_tot = len(starts)
                print(f"  {method}: {n_ok}/{n_tot} converged")
    return results
