import numpy as np


def is_convex_at(hess_fn, x, tol=-1e-6):
    """Return True if H(x) is positive semi-definite within tolerance tol."""
    H = np.asarray(hess_fn(x), dtype=float)
    return float(np.linalg.eigvalsh(0.5 * (H + H.T)).min()) >= tol


def estimate_L3(hess_fn, x0, eps=1e-4, n_dirs=10):
    """
    Estimate the Hessian Lipschitz constant L3 via finite differences at x0.

    Computes max over random unit directions v of ||H(x0 + eps*v) - H(x0)||_F / eps.
    """
    H0 = hess_fn(x0)
    rng = np.random.default_rng(0)
    L3 = 0.0
    for _ in range(n_dirs):
        v = rng.normal(size=len(x0))
        v /= np.linalg.norm(v)
        H1 = hess_fn(x0 + eps * v)
        L3 = max(L3, np.linalg.norm(H1 - H0, "fro") / eps)
    return max(L3, 1e-6)


def summarize(solver):
    if not hasattr(solver, "log"):
        raise AttributeError("Solver has no attribute 'log'. Did you run it?")
    if len(solver.log) == 0:
        raise ValueError("Solver log is empty. Did the solver run?")

    log = solver.log
    return {
        "iterations": len(log),
        "runtime": solver.runtime,
        "final_f": log[-1]["f"],
        "final_grad_norm": log[-1]["grad_norm"],
        "rejected_steps": sum(not e["accepted"] for e in log),
    }
