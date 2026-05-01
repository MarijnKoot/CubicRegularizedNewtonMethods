"""
benchmarks.py — Problem suite and defaults for the large-scale synthesis experiment.

Defaults carried over from earlier experiments:
  EPS_G    = 1e-7   (experiments/2 and 3)
  MAX_ITER = 500    (all earlier experiments)
  SEED     = 42     (all earlier experiments)
  sigma0   = max(2*L3, 0.5)  (adapters.py default for NCR/ARC)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from harness.problems import get_problem

# ── shared defaults ────────────────────────────────────────────────────────────
SEED     = 42
EPS_G    = 1e-7
MAX_ITER = 500
METHODS  = ["Newton", "NCR", "ARC", "ACRN"]


# ── benchmark instance ─────────────────────────────────────────────────────────

@dataclass
class Instance:
    family:        str
    problem:       str
    label:         str
    n:             int
    problem_kwargs: dict
    x0s:           Dict[str, np.ndarray]  # label -> starting point


def _std_x0s(n: int, seed: int, standard: np.ndarray, benign: np.ndarray) -> Dict[str, np.ndarray]:
    return {"standard": standard.copy(), "benign": benign.copy()}


# ── convex family: LogSumExp ───────────────────────────────────────────────────

def make_logsumexp() -> List[Instance]:
    instances = []
    for n in [10, 20, 50]:
        rng = np.random.default_rng(SEED + 1000)
        x0  = rng.standard_normal(n)
        instances.append(Instance(
            family="logsumexp",
            problem="logsumexp",
            label=f"LogSumExp n={n}",
            n=n,
            problem_kwargs={},
            # benign = small random (zeros causes trivial gradient for logsumexp)
            x0s=_std_x0s(n, SEED, standard=x0, benign=rng.standard_normal(n) * 0.1),
        ))
    return instances


# ── nonconvex family: Rosenbrock ───────────────────────────────────────────────

def make_rosenbrock() -> List[Instance]:
    instances = []
    for n in [2, 10, 20, 50]:
        standard = np.full(n, -1.0)
        benign   = np.full(n,  0.5)
        instances.append(Instance(
            family="rosenbrock",
            problem="rosenbrock",
            label=f"Rosenbrock n={n}",
            n=n,
            problem_kwargs={},
            x0s=_std_x0s(n, SEED, standard=standard, benign=benign),
        ))
    return instances


# ── ill-conditioned convex family: quadratic ───────────────────────────────────

def make_ill_conditioned() -> List[Instance]:
    instances = []
    rng = np.random.default_rng(SEED)
    for n in [10, 20, 50]:
        x0 = rng.standard_normal(n)
        for cond in [1e2, 1e4, 1e6]:
            instances.append(Instance(
                family="ill_conditioned",
                problem="quadratic",
                label=f"Quadratic n={n} κ={cond:.0e}",
                n=n,
                problem_kwargs={"cond": cond},
                x0s=_std_x0s(n, SEED, standard=x0, benign=np.zeros(n)),
            ))
    return instances


ALL_INSTANCES: List[Instance] = (
    make_logsumexp()
    + make_rosenbrock()
    + make_ill_conditioned()
)
