"""
harness/problems.py — Problem registry for benchmark experiments.

get_problem(name, n, seed=42, problem_kwargs=None)
    Returns (f, grad, hess, meta) where meta contains at least:
        "L3"     : float | None  — Hessian Lipschitz constant (None = unknown)
        "convex" : bool

Registered problems
-------------------
quadratic          — SPD quadratic 0.5 x^T A x - b^T x
                     kwargs: cond (condition number, default 1.0)
near_psd_quadratic — quadratic with prescribed minimum eigenvalue
                     kwargs: lambda_min (default 1e-4)
cubic_norm         — (1/3)||x||^3
rosenbrock         — Rosenbrock chain
logsumexp          — log-sum-exp with random data (m=2n, mu=1)
rastrigin          — Rastrigin (A=10)
dixon_price        — Dixon-Price
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def get_problem(
    name: str,
    n: int,
    seed: int = 42,
    problem_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple:
    """Return (f, grad, hess, meta)."""
    if problem_kwargs is None:
        problem_kwargs = {}
    registry = {
        "quadratic":           _quadratic,
        "near_psd_quadratic":  _near_psd_quadratic,
        "cubic_norm":          _cubic_norm,
        "quartic_convex":      _quartic_convex,
        "rosenbrock":          _rosenbrock,
        "logsumexp":           _logsumexp,
        "rastrigin":           _rastrigin,
        "dixon_price":         _dixon_price,
    }
    if name not in registry:
        raise ValueError(f"Unknown problem {name!r}. Available: {list(registry)}")
    return registry[name](n, seed, **problem_kwargs)


# ── individual factories ───────────────────────────────────────────────────────

def _quadratic(n: int, seed: int, cond: float = 1.0):
    from test_functions.quadratic import quadratic
    rng = np.random.default_rng(seed)
    if cond <= 1.0:
        M = rng.standard_normal((n, n))
        A = M.T @ M + np.eye(n)
    else:
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        lam = np.exp(np.linspace(0.0, np.log(float(cond)), n))
        A = Q @ np.diag(lam) @ Q.T
        A = 0.5 * (A + A.T)
    b = np.zeros(n)
    f, grad, hess, _ = quadratic(A, b)
    return f, grad, hess, {"L3": 0.0, "convex": True}


def _near_psd_quadratic(n: int, seed: int, lambda_min: float = 1e-4):
    from test_functions.quadratic import quadratic
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.linspace(float(max(lambda_min, 0.0)), 1.0, n)
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    b = np.zeros(n)
    f, grad, hess, _ = quadratic(A, b)
    convex = lambda_min >= 0.0
    return f, grad, hess, {"L3": 0.0, "convex": convex}


def _cubic_norm(n: int, seed: int):
    from test_functions.cubic_norm import cubic_norm
    f, grad, hess, L3 = cubic_norm(n)
    return f, grad, hess, {"L3": float(L3), "convex": True}


def _quartic_convex(n: int, seed: int):
    from test_functions.quartic_convex import quartic_convex
    f, grad, hess = quartic_convex(n)
    return f, grad, hess, {"L3": None, "convex": True}


def _rosenbrock(n: int, seed: int):
    from test_functions.rosenbrock import rosenbrock
    f, grad, hess = rosenbrock(n)
    return f, grad, hess, {"L3": None, "convex": False}


def _logsumexp(n: int, seed: int):
    from test_functions.logsumexp import logsumexp
    rng = np.random.default_rng(seed)
    m   = 2 * n
    mu  = 1.0
    a   = rng.standard_normal((m, n))
    b   = rng.standard_normal(m)
    f, grad, hess, M = logsumexp(a, b, mu)
    return f, grad, hess, {"L3": None, "convex": True}


def _rastrigin(n: int, seed: int):
    from test_functions.rastrigin import rastrigin
    f, grad, hess = rastrigin(n)
    return f, grad, hess, {"L3": None, "convex": False}


def _dixon_price(n: int, seed: int):
    from test_functions.dixon_price import dixon_price
    f, grad, hess = dixon_price(n)
    return f, grad, hess, {"L3": None, "convex": False}
