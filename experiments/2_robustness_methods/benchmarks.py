"""
benchmarks.py — Benchmark suite definitions for the robustness experiment.

Each benchmark instance is a named tuple describing one (problem, geometry, start) triple.
The `get_instance` function returns (f, grad, hess, meta, x0_dict) ready for running.

Families
--------
1. ill_conditioned  — SPD quadratics with condition numbers [1e2 .. 1e8] + near-singular + singular
2. rosenbrock       — Rosenbrock chain, n = 2, 10, 20
3. dixon_price      — Dixon-Price, n = 10, 20, 50
4. rastrigin        — Rastrigin, n = 10, 20, 50
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from harness.problems import get_problem

SEED      = 42
EPS_G     = 1e-7
MAX_ITER  = 500
N_RANDOM  = 8      # random perturbations added on top of standard + benign
PERTURB   = 2.0    # std of Gaussian noise added to standard start


# ── benchmark instance descriptor ─────────────────────────────────────────────

@dataclass
class BenchmarkInstance:
    family:      str    # "ill_conditioned" | "rosenbrock" | "dixon_price" | "rastrigin"
    problem:     str    # registry name
    label:       str    # human-readable (e.g. "Quadratic κ=1e4")
    n:           int
    problem_kwargs: dict
    x0s: Dict[str, np.ndarray]   # label -> starting point


# ── shared starting-point generator ───────────────────────────────────────────

def _make_x0s(
    standard: np.ndarray,
    benign:   np.ndarray,
    perturb:  float = PERTURB,
    n_random: int   = N_RANDOM,
) -> Dict[str, np.ndarray]:
    """
    Returns 2 + n_random starting points:
      "standard"  — the fixed benchmark start
      "benign"    — a milder start (closer to optimum)
      "random_0" … "random_{n_random-1}"  — standard + N(0, perturb²) noise
    """
    rng = np.random.default_rng(SEED)
    x0s: Dict[str, np.ndarray] = {
        "standard": standard.copy(),
        "benign":   benign.copy(),
    }
    for i in range(n_random):
        x0s[f"random_{i}"] = standard + rng.standard_normal(len(standard)) * perturb
    return x0s


def _rosenbrock_x0(n: int) -> Dict[str, np.ndarray]:
    return _make_x0s(
        standard=np.full(n, -1.0),
        benign=np.full(n, 0.5),
    )


def _general_x0(n: int, standard: np.ndarray, benign: np.ndarray) -> Dict[str, np.ndarray]:
    return _make_x0s(standard=standard, benign=benign)


def _rastrigin_x0(n: int) -> Dict[str, np.ndarray]:
    """Starts inside the first basin of attraction of Rastrigin."""
    rng = np.random.default_rng(SEED + 1)
    return _make_x0s(
        standard=rng.uniform(-2.0, 2.0, n),
        benign=rng.uniform(-0.5, 0.5, n),
        perturb=1.0,   # smaller perturbation to stay near first basin
    )


# ── benchmark lists ───────────────────────────────────────────────────────────

def make_ill_conditioned() -> List[BenchmarkInstance]:
    instances = []
    n = 10
    rng = np.random.default_rng(SEED)
    standard = rng.standard_normal(n)
    for cond in [1e2, 1e4, 1e6, 1e8]:
        instances.append(BenchmarkInstance(
            family="ill_conditioned",
            problem="quadratic",
            label=f"Quadratic $\\kappa$={cond:.0e}",
            n=n,
            problem_kwargs={"cond": cond},
            x0s=_make_x0s(standard=standard, benign=np.zeros(n)),
        ))
    for lm in [1e-4, 1e-6, 0.0]:
        label = f"NearPSD $\\lambda_{{\\min}}$={lm:.0e}" if lm > 0 else "NearPSD (singular)"
        instances.append(BenchmarkInstance(
            family="ill_conditioned",
            problem="near_psd_quadratic",
            label=label,
            n=n,
            problem_kwargs={"lambda_min": lm},
            x0s=_make_x0s(standard=standard, benign=np.zeros(n)),
        ))
    return instances


def make_rosenbrock() -> List[BenchmarkInstance]:
    return [
        BenchmarkInstance(
            family="rosenbrock",
            problem="rosenbrock",
            label=f"Rosenbrock $n$={n}",
            n=n,
            problem_kwargs={},
            x0s=_rosenbrock_x0(n),
        )
        for n in [2, 10, 20]
    ]


def make_dixon_price() -> List[BenchmarkInstance]:
    return [
        BenchmarkInstance(
            family="dixon_price",
            problem="dixon_price",
            label=f"Dixon-Price $n$={n}",
            n=n,
            problem_kwargs={},
            x0s=_general_x0(n, standard=np.ones(n), benign=np.ones(n) * 0.1),
        )
        for n in [10, 20, 50]
    ]


def make_rastrigin() -> List[BenchmarkInstance]:
    return [
        BenchmarkInstance(
            family="rastrigin",
            problem="rastrigin",
            label=f"Rastrigin $n$={n}",
            n=n,
            problem_kwargs={},
            x0s=_rastrigin_x0(n),
        )
        for n in [10, 20, 50]
    ]


ALL_BENCHMARKS: List[BenchmarkInstance] = (
    make_ill_conditioned()
    + make_rosenbrock()
    + make_dixon_price()
    + make_rastrigin()
)

METHODS = ["Newton", "NCR", "ARC", "ACRN"]
