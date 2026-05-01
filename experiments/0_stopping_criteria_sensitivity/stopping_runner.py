"""
stopping_runner.py — core logic for the stopping-criteria sensitivity experiment.

Strategy: run each solver once to the tightest tolerance (eps_g = 1e-9),
collecting the full per-iteration history. Then replay that history offline
to find the first iteration that satisfies each (stopping_mode, eps_g) combo.
This avoids re-running the solver N_eps times per configuration.

Stopping modes
--------------
grad_only   : stop when  ||g_k|| <= eps_g
              (plus max-iter safeguard, already handled by the solver)

grad_step   : stop when  ||g_k|| <= eps_g
                      OR ||x_{k+1} - x_k|| <= eps_x * (1 + ||x_k||)
                      OR |f_{k+1} - f_k|   <= eps_f * (1 + |f_k|)
              (plus max-iter safeguard)

eps_x = 1e-10  (fixed)
eps_f = 1e-12  (fixed)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
EPS_X  = 1e-10   # relative step-size threshold
EPS_F  = 1e-12   # relative function-change threshold
TIGHT  = 1e-9    # tolerance used for the actual solver run

METHODS  = ["Newton", "NCR", "ARC", "ACRN"]
EPS_GRID = [1e-3, 1e-5, 1e-7, 1e-9]
MODES    = ["grad_only", "grad_step"]

MAX_ITER = 500
SEED     = 42


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StoppingResult:
    problem:        str
    method:         str
    stopping_mode:  str
    eps_g:          float
    eps_x:          float
    eps_f:          float
    success:        bool      # final_grad_norm <= eps_g
    termination_reason: str   # "grad_tol" | "step_tol" | "f_tol" | "max_iter" | "error:…"
    iterations:     int       # first iteration satisfying the stopping rule
    runtime_sec:    float     # wall-clock time of the full solver run (shared across eps_g)
    final_f:        float
    final_grad_norm: float
    final_x:        str       # serialised as "v0,v1,…" (short if small)
    rejected_iter:  int       # rejected / inner trials (0 for Newton)

    def to_row(self) -> dict:
        return {
            "problem":             self.problem,
            "method":              self.method,
            "stopping_mode":       self.stopping_mode,
            "eps_g":               self.eps_g,
            "eps_x":               self.eps_x,
            "eps_f":               self.eps_f,
            "success":             int(self.success),
            "termination_reason":  self.termination_reason,
            "iterations":          self.iterations,
            "runtime_sec":         self.runtime_sec,
            "final_f":             self.final_f,
            "final_grad_norm":     self.final_grad_norm,
            "final_x":             self.final_x,
            "rejected_iter":       self.rejected_iter,
        }

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "problem", "method", "stopping_mode",
            "eps_g", "eps_x", "eps_f",
            "success", "termination_reason",
            "iterations", "runtime_sec",
            "final_f", "final_grad_norm",
            "final_x", "rejected_iter",
        ]


# ── per-run logic ─────────────────────────────────────────────────────────────

def _x0_for(problem: str, n: int, seed: int = SEED) -> np.ndarray:
    """Starting point for a given problem, derived from seed."""
    if problem == "rosenbrock":
        return np.full(n, -1.0)        # classic hard start; seed doesn't vary this
    rng = np.random.default_rng(seed + 1000)   # offset to differ from problem seed
    return rng.standard_normal(n)


def _replay_history(
    log: List[dict],
    grad_fn,
    f_fn,
    eps_g: float,
    mode: str,
) -> Tuple[int, str, float, float, np.ndarray]:
    """
    Walk through the solver log and find the first iteration satisfying
    the stopping criterion defined by (mode, eps_g).

    Returns (iteration_index, reason, final_f, final_grad_norm, final_x).
    iteration_index is 1-based (number of steps taken).
    """
    n_log = len(log)

    for k, entry in enumerate(log):
        g_norm     = float(entry["grad_norm"])
        x_k        = np.asarray(entry["x"], dtype=float)
        is_sentinel = entry.get("is_sentinel", False)
        # Sentinel is the post-step state of the final iteration, so it counts
        # as the same iteration as the previous entry (k steps were taken).
        iters = k if is_sentinel else k + 1

        # Gradient criterion (applies to both modes)
        if g_norm <= eps_g:
            f_val = float(f_fn(x_k))
            return iters, "grad_tol", f_val, g_norm, x_k

        if mode == "grad_step" and k > 0:
            prev   = log[k - 1]
            x_prev = np.asarray(prev["x"], dtype=float)
            f_prev = float(prev["f"])

            step_norm    = float(norm(x_k - x_prev))
            rel_step     = step_norm / (1.0 + float(norm(x_prev)))
            f_k          = float(entry["f"])
            rel_f_change = abs(f_k - f_prev) / (1.0 + abs(f_prev))

            if rel_step <= EPS_X:
                return iters, "step_tol", float(f_fn(x_k)), float(norm(grad_fn(x_k))), x_k
            if rel_f_change <= EPS_F:
                return iters, "f_tol", float(f_fn(x_k)), float(norm(grad_fn(x_k))), x_k

    # Exhausted log: report at the last available point
    last  = log[-1]
    x_end = np.asarray(last["x"], dtype=float)
    f_end = float(f_fn(x_end))
    g_end = float(norm(grad_fn(x_end)))
    return n_log, "max_iter", f_end, g_end, x_end


def run_problem_method(
    problem_name: str,
    n:            int,
    method:       str,
    verbose:      bool = True,
    seed:         int  = SEED,
) -> List[StoppingResult]:
    """
    Run one (problem, method) pair once at TIGHT tolerance,
    then derive results for every (mode, eps_g) combination.
    """
    f_raw, grad_raw, hess_raw, meta = get_problem(
        problem_name, n, seed=seed
    )
    counter = EvalCounter(f_raw, grad_raw, hess_raw)
    L3 = meta.get("L3")

    # ACRN requires convex problem
    if method == "ACRN" and not meta.get("convex", False):
        reason = "error:ACRN requires convex problem"
        results = []
        for mode in MODES:
            for eps_g in EPS_GRID:
                results.append(StoppingResult(
                    problem=problem_name, method=method,
                    stopping_mode=mode, eps_g=eps_g,
                    eps_x=EPS_X, eps_f=EPS_F,
                    success=False, termination_reason=reason,
                    iterations=0, runtime_sec=float("nan"),
                    final_f=float("nan"), final_grad_norm=float("nan"),
                    final_x="", rejected_iter=0,
                ))
        return results

    x0 = _x0_for(problem_name, n, seed=seed)

    try:
        raw = run_and_extract(
            method=method,
            counter=counter,
            x0=x0.copy(),
            eps_g=TIGHT,
            max_iter=MAX_ITER,
            L3=L3,
            solver_kwargs={},
        )
    except Exception as exc:
        reason = f"error:{str(exc)[:120]}"
        results = []
        for mode in MODES:
            for eps_g in EPS_GRID:
                results.append(StoppingResult(
                    problem=problem_name, method=method,
                    stopping_mode=mode, eps_g=eps_g,
                    eps_x=EPS_X, eps_f=EPS_F,
                    success=False, termination_reason=reason,
                    iterations=0, runtime_sec=float("nan"),
                    final_f=float("nan"), final_grad_norm=float("nan"),
                    final_x="", rejected_iter=0,
                ))
        return results

    elapsed      = raw["elapsed"]
    rejected     = raw["rejected_iterations"]
    grad_history = raw["grad_norm_history"]
    f_history    = raw["f_history"]

    # Reconstruct minimal log from histories (x-history requires re-running the
    # solver's internal log; we access it via the solver object stored in raw
    # if available, otherwise fall back to the history arrays).
    # The adapters do not return the raw solver object, but they return
    # grad_norm_history and f_history. For x we need the full log.
    # We re-run with a lightweight wrapper that captures the log.
    # Rather than doing a second run, we call _run_and_get_log() which runs once.
    if verbose:
        print(f"  {problem_name} / {method} ...", end=" ", flush=True)

    log = _run_and_get_log(method, f_raw, grad_raw, hess_raw, x0, L3, seed=seed)
    elapsed_log = log["elapsed"]

    results = []
    for mode in MODES:
        for eps_g in EPS_GRID:
            iters, reason, final_f, final_gnorm, final_x = _replay_history(
                log["entries"], grad_raw, f_raw, eps_g, mode
            )
            success = final_gnorm <= eps_g
            x_str   = ",".join(f"{v:.6e}" for v in final_x[:10])  # first 10 components
            if len(final_x) > 10:
                x_str += f",…(len={len(final_x)})"
            results.append(StoppingResult(
                problem=problem_name, method=method,
                stopping_mode=mode, eps_g=eps_g,
                eps_x=EPS_X, eps_f=EPS_F,
                success=success, termination_reason=reason,
                iterations=iters, runtime_sec=elapsed_log,
                final_f=final_f, final_grad_norm=final_gnorm,
                final_x=x_str, rejected_iter=rejected,
            ))

    if verbose:
        n_ok = sum(1 for r in results if r.success)
        print(f"done  ({n_ok}/{len(results)} combos converged)")

    return results


def _run_and_get_log(
    method: str,
    f_raw, grad_raw, hess_raw,
    x0: np.ndarray,
    L3: Optional[float],
    seed: int = SEED,
) -> dict:
    """
    Run the solver at TIGHT tolerance and return its internal log list
    together with elapsed time.

    Each entry in log["entries"] has at minimum: grad_norm, f, x, accepted.
    """
    counter = EvalCounter(f_raw, grad_raw, hess_raw)

    # Import solvers directly to access .log
    from methods.pure_newton import PureNewton, PureNewtonOptions
    from methods.NCR         import CubicNewton, CRNOptions
    from methods.ARC         import AdaptiveCubicNewton, ARCParams
    from methods.ACRN        import AcceleratedCubicNewton

    t0 = time.perf_counter()

    if method == "Newton":
        opts    = PureNewtonOptions(tol_grad=TIGHT, max_iter=MAX_ITER)
        solver  = PureNewton(counter.f, counter.grad, counter.hess, options=opts)
        x_final = solver.run(x0.copy())
        log_raw = solver.log

    elif method == "NCR":
        sigma_min = CRNOptions().sigma_min
        opts    = CRNOptions(sigma0=sigma_min, sigma_min=sigma_min,
                             tol_grad=TIGHT, max_iter=MAX_ITER)
        solver  = CubicNewton(counter.f, counter.grad, counter.hess, options=opts)
        x_final = solver.run(x0.copy())
        log_raw = solver.log

    elif method == "ARC":
        sigma_min = ARCParams().sigma_min
        params  = ARCParams(sigma0=sigma_min, sigma_min=sigma_min,
                            tol_grad=TIGHT, max_iter=MAX_ITER)
        solver  = AdaptiveCubicNewton(
            counter.f, counter.grad, counter.hess,
            params=params, step_method="secular",
        )
        x_final = solver.run(x0.copy())
        log_raw = solver.log

    elif method == "ACRN":
        L3_eff    = L3 if (L3 is not None and L3 > 0) else max(estimate_L3(hess_raw, x0), 1e-15)
        sigma_min = 1e-15
        solver  = AcceleratedCubicNewton(
            counter.f, counter.grad, counter.hess,
            L3=L3_eff, sigma=sigma_min, sigma_min=sigma_min,
            tol_grad=TIGHT, max_iter=MAX_ITER, adaptive_sigma=True,
        )
        x_final, _ = solver.run(x0.copy())
        log_raw = solver.log

    else:
        raise ValueError(f"Unknown method: {method!r}")

    elapsed = time.perf_counter() - t0

    # Normalise log: ensure each entry has grad_norm, f, x
    entries = []
    for entry in log_raw:
        entries.append({
            "grad_norm": float(entry.get("grad_norm", float("nan"))),
            "f":         float(entry.get("f",         float("nan"))),
            "x":         np.asarray(entry.get("x", x0), dtype=float).copy(),
            "accepted":  bool(entry.get("accepted", True)),
        })

    # Append a sentinel entry at x_final. Solvers log the state at x_k before
    # taking the step, so the converged point x_final is never in the log.
    # Without this, _replay_history can never observe the satisfied criterion.
    # is_sentinel=True tells _replay_history not to count this as a new iteration.
    x_final = np.asarray(x_final, dtype=float)
    g_final = float(norm(grad_raw(x_final)))
    f_final = float(f_raw(x_final))
    entries.append({
        "grad_norm":   g_final,
        "f":           f_final,
        "x":           x_final.copy(),
        "accepted":    True,
        "is_sentinel": True,
    })

    return {"entries": entries, "elapsed": elapsed}
