"""
harness/adapters.py — Unified solver runner returning a standardised result dict.

run_and_extract(method, counter, x0, eps_g, max_iter, L3, solver_kwargs={})
    Returns dict with keys:
        x_final, stop_reason, outer_iterations, rejected_iterations,
        sigma0, sigma_final, elapsed, grad_norm_history, f_history
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.linalg import norm

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from utilities import estimate_L3


def _sigma0_from_L3(L3: Optional[float]) -> float:
    if L3 is None:
        return 1.0
    if L3 <= 0.0:
        return 0.5
    return max(2.0 * float(L3), 0.5)


def run_and_extract(
    method: str,
    counter,
    x0: np.ndarray,
    eps_g: float,
    max_iter: int,
    L3: Optional[float],
    solver_kwargs: Optional[Dict[str, Any]] = None,
) -> dict:
    if solver_kwargs is None:
        solver_kwargs = {}

    from methods.pure_newton import PureNewton, PureNewtonOptions
    from methods.NCR import CubicNewton, CRNOptions
    from methods.ARC import AdaptiveCubicNewton, ARCParams
    from methods.ACRN import AcceleratedCubicNewton

    t0 = time.perf_counter()

    if method == "Newton":
        opts = PureNewtonOptions(tol_grad=eps_g, max_iter=max_iter,
                                 **solver_kwargs)
        solver = PureNewton(counter.f, counter.grad, counter.hess, options=opts)
        x_final = solver.run(x0.copy())
        log = solver.log
        reason = solver.termination_reason or "max_iter"
        sigma0_used = float("nan")
        sigma_final = float("nan")
        rejected = 0

    elif method == "NCR":
        s0 = max(_sigma0_from_L3(L3), 1e-12)
        opts = CRNOptions(sigma0=s0, tol_grad=eps_g, max_iter=max_iter,
                          **solver_kwargs)
        solver = CubicNewton(counter.f, counter.grad, counter.hess, options=opts)
        x_final = solver.run(x0.copy())
        log = solver.log
        reason = solver.termination_reason or "max_iter"
        sigma0_used = s0
        sigma_final = float(solver.sigma) if hasattr(solver, "sigma") else float("nan")
        rejected = sum(1 for e in log if not e.get("accepted", True))

    elif method == "ARC":
        s0 = max(_sigma0_from_L3(L3), 1e-12)
        params = ARCParams(sigma0=s0, tol_grad=eps_g, max_iter=max_iter,
                           **solver_kwargs)
        solver = AdaptiveCubicNewton(counter.f, counter.grad, counter.hess,
                                     params=params, step_method="secular")
        x_final = solver.run(x0.copy())
        log = solver.log
        reason = solver.termination_reason or "max_iter"
        sigma0_used = s0
        last_sigma = log[-1]["sigma"] if log else s0
        sigma_final = float(last_sigma)
        rejected = sum(1 for e in log if not e.get("accepted", True))

    elif method == "ACRN":
        if L3 is None:
            L3_eff = float(estimate_L3(counter.hess, x0))
        else:
            L3_eff = float(L3)
        s0 = max(L3_eff, 1e-12)
        solver = AcceleratedCubicNewton(
            counter.f, counter.grad, counter.hess,
            L3=s0, sigma=s0,
            sigma_min=1e-15, sigma_max=5e11,
            tol_grad=eps_g, max_iter=max_iter,
            adaptive_sigma=True,
            **solver_kwargs,
        )
        result = solver.run(x0.copy())
        x_final = result[0] if isinstance(result, tuple) else result
        log = solver.log
        reason = solver.termination_reason or "max_iter"
        sigma0_used = s0
        last_sigma = log[-1].get("sigma_k", s0) if log else s0
        sigma_final = float(last_sigma)
        rejected = sum(int(e.get("rejected_trials", 0)) for e in log)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    elapsed = time.perf_counter() - t0
    x_final = np.asarray(x_final, dtype=float)

    grad_norm_history = [float(e.get("grad_norm", float("nan"))) for e in log]
    f_history         = [float(e.get("f",         float("nan"))) for e in log]

    return {
        "x_final":             x_final,
        "stop_reason":         reason,
        "outer_iterations":    len(log),
        "rejected_iterations": rejected,
        "sigma0":              sigma0_used,
        "sigma_final":         sigma_final,
        "elapsed":             elapsed,
        "grad_norm_history":   grad_norm_history,
        "f_history":           f_history,
    }
