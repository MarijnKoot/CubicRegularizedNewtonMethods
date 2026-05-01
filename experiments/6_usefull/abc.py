"""
CUTEst benchmark runner for the Appendix A problem set.

This script benchmarks the three local solvers on a curated list of
unconstrained CUTEst problems matching the Appendix A names from the paper.
It writes a long-form CSV with one row per (problem, solver) run.

Run from the repo root inside Ubuntu/WSL, for example:

    source venv/bin/activate
    python experiments/benchmark_cutest_appendix_a.py --limit 5
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

PYCUTEST_CACHE = Path(os.environ.setdefault("PYCUTEST_CACHE", str(Path.home() / ".pycutest_cache")))
PYCUTEST_CACHE.mkdir(parents=True, exist_ok=True)
PYCUTEST_CACHE_HOLDER = PYCUTEST_CACHE / "pycutest_cache_holder"
PYCUTEST_CACHE_HOLDER.mkdir(parents=True, exist_ok=True)
(PYCUTEST_CACHE_HOLDER / "__init__.py").touch(exist_ok=True)
if str(PYCUTEST_CACHE) not in sys.path:
    sys.path.insert(0, str(PYCUTEST_CACHE))

import pycutest

from methods import ACRN as acrn_module
from methods import ARC as arc_module
from methods import NCR as ncr_module

CONVEXITY_TOL = 1e-6
L3_EPS = 1e-4
L3_N_DIRS = 3


APPENDIX_A_PROBLEMS: List[Dict[str, object]] = [
    {"name": "ALLINITU", "n": 4},
    {"name": "ARGLINA", "n": 200},
    {"name": "ARWHEAD", "n": 100},
    {"name": "BARD", "n": 3},
    {"name": "BDQRTIC", "n": 100},
    {"name": "BEALE", "n": 2},
    {"name": "BIGGS6", "n": 6},
    {"name": "BOX3", "n": 3},
    {"name": "BRKMCC", "n": 2},
    {"name": "BROWNAL", "n": 200},
    {"name": "BROWNBS", "n": 2},
    {"name": "BROWNDEN", "n": 4},
    {"name": "BROYDN7D", "n": 100},
    {"name": "BRYBND", "n": 100},
    {"name": "CHAINWOO", "n": 100},
    {"name": "CHNROSNB", "n": 50},
    {"name": "CLIFF", "n": 2},
    {"name": "CRAGGLVY", "n": 202},
    {"name": "CUBE", "n": 2},
    {"name": "CURLY10", "n": 50},
    {"name": "CURLY20", "n": 50},
    {"name": "CURLY30", "n": 50},
    {"name": "DECONVU", "n": 61},
    {"name": "DENSCHNA", "n": 2},
    {"name": "DENSCHNB", "n": 2},
    {"name": "DENSCHNC", "n": 2},
    {"name": "DENSCHND", "n": 3},
    {"name": "DENSCHNE", "n": 3},
    {"name": "DENSCHNF", "n": 2},
    {"name": "DIXMAANA", "n": 150},
    {"name": "DIXMAANB", "n": 150},
    {"name": "DIXMAANC", "n": 150},
    {"name": "DIXMAAND", "n": 150},
    {"name": "DIXMAANE", "n": 150},
    {"name": "DIXMAANF", "n": 150},
    {"name": "DIXMAANG", "n": 150},
    {"name": "DIXMAANH", "n": 150},
    {"name": "DIXMAANI", "n": 150},
    {"name": "DIXMAANJ", "n": 150},
    {"name": "DIXMAANK", "n": 150},
    {"name": "DIXMAANL", "n": 150},
    {"name": "DJTL", "n": 2},
    {"name": "DQRTIC", "n": 100},
    {"name": "EDENSCH", "n": 100},
    {"name": "EG2", "n": 100},
    {"name": "EIGENALS", "n": 110},
    {"name": "EIGENBLS", "n": 110},
    {"name": "EIGENCLS", "n": 132},
    {"name": "ENGVAL1", "n": 100},
    {"name": "ENGVAL2", "n": 3},
    {"name": "ERRINROS", "n": 50},
    {"name": "EXPFIT", "n": 2},
    {"name": "EXTROSNB", "n": 100},
    {"name": "FLETCBV2", "n": 100},
    {"name": "FLETCBV3", "n": 50},
    {"name": "FLETCHBV", "n": 10},
    {"name": "FLETCHCR", "n": 100},
    {"name": "FMINSRF2", "n": 121},
    {"name": "FMINSURF", "n": 121},
    {"name": "FREUROTH", "n": 100},
    {"name": "GENHUMPS", "n": 10},
    {"name": "GENROSE", "n": 100},
    {"name": "GENROSEB", "n": 500},
    {"name": "GROWTHLS", "n": 3},
    {"name": "GULF", "n": 3},
    {"name": "HAIRY", "n": 2},
    {"name": "HATFLDD", "n": 3},
    {"name": "HATFLDE", "n": 3},
    {"name": "HEART6LS", "n": 6},
    {"name": "HEART8LS", "n": 8},
    {"name": "HELIX", "n": 3},
    {"name": "HIMMELBB", "n": 2},
    {"name": "HUMPS", "n": 2},
    {"name": "HYDC20LS", "n": 99},
    {"name": "JENSMP", "n": 2},
    {"name": "KOWOSB", "n": 4},
    {"name": "LIARWHD", "n": 100},
    {"name": "LOGHAIRY", "n": 2},
    {"name": "MANCINO", "n": 100},
    {"name": "MEXHAT", "n": 2},
    {"name": "MEYER3", "n": 3},
    {"name": "MOREBV", "n": 100},
    {"name": "MSQRTALS", "n": 100},
    {"name": "MSQRTBLS", "n": 100},
    {"name": "NONCVXU2", "n": 100},
    {"name": "NONCVXUN", "n": 100},
    {"name": "NONDIA", "n": 100},
    {"name": "NONDQUAR", "n": 100},
    {"name": "NONMSQRT", "n": 100},
    {"name": "OSBORNEA", "n": 5},
    {"name": "OSBORNEB", "n": 11},
    {"name": "OSCIPATH", "n": 8},
    {"name": "PALMER5C", "n": 6},
    {"name": "PALMER6C", "n": 8},
    {"name": "PALMER7C", "n": 8},
    {"name": "PALMER8C", "n": 8},
    {"name": "PARKCH", "n": 15},
    {"name": "PENALTY1", "n": 100},
    {"name": "PENALTY2", "n": 200},
    {"name": "PENALTY3", "n": 200},
    {"name": "PFIT1LS", "n": 3},
    {"name": "PFIT2LS", "n": 3},
    {"name": "PFIT3LS", "n": 3},
    {"name": "PFIT4LS", "n": 3},
    {"name": "POWELLSG", "n": 4},
    {"name": "POWER", "n": 100},
    {"name": "QUARTC", "n": 100},
    {"name": "ROSENBR", "n": 2},
    {"name": "S308", "n": 2},
    {"name": "SBRYBND", "n": 100},
    {"name": "SCHMVETT", "n": 100},
    {"name": "SENSORS", "n": 100},
    {"name": "SINEVAL", "n": 2},
    {"name": "SINQUAD", "n": 100},
    {"name": "SISSER", "n": 2},
    {"name": "SNAIL", "n": 2},
    {"name": "SPARSINE", "n": 100},
    {"name": "SPARSQUR", "n": 100},
    {"name": "SPMSRTLS", "n": 100},
    {"name": "SROSENBR", "n": 100},
    {"name": "STREG", "n": 4},
    {"name": "TOINTGOR", "n": 50},
    {"name": "TOINTGSS", "n": 100},
    {"name": "TOINTPSP", "n": 50},
    {"name": "TQUARTIC", "n": 100},
    {"name": "VARDIM", "n": 200},
    {"name": "VAREIGVL", "n": 50},
    {"name": "VIBRBEAM", "n": 8},
    {"name": "WATSON", "n": 12},
    {"name": "WOODS", "n": 4},
    {"name": "YFITU", "n": 3},
]


SPECIAL_SIF_PARAMS: Dict[str, Dict[str, int]] = {
    # The generic {"N": n} fallback covers many parameterized problems.
    # This map is for problems where the target dimension is known to need
    # a nontrivial parameterization.
    "ARGLALE": {"N": 100, "M": 200},
}


@dataclass
class EvalCounter:
    f_calls: int = 0
    g_calls: int = 0
    h_calls: int = 0


class ProblemAdapter:
    def __init__(self, problem) -> None:
        self.problem = problem
        self.counter = EvalCounter()

    def f(self, x: np.ndarray) -> float:
        self.counter.f_calls += 1
        return float(self.problem.obj(np.asarray(x, dtype=float)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.counter.g_calls += 1
        _, g = self.problem.obj(np.asarray(x, dtype=float), gradient=True)
        return np.asarray(g, dtype=float)

    def hess(self, x: np.ndarray) -> np.ndarray:
        self.counter.h_calls += 1
        return np.asarray(self.problem.ihess(np.asarray(x, dtype=float)), dtype=float)


def check_convexity(problem, properties: Dict[str, object]) -> tuple[bool, float]:
    """
    Returns (is_convex, min_eigenvalue_at_x0).

    Layer 1: 'sum of squares' objective is always convex.
    Layer 2: minimum eigenvalue of symmetrised H(x0) >= -CONVEXITY_TOL.
    """
    H = np.asarray(problem.ihess(problem.x0), dtype=float)
    H = 0.5 * (H + H.T)
    min_eig = float(np.linalg.eigvalsh(H).min())
    if properties.get("objective") == "sum of squares":
        return True, min_eig
    return min_eig >= -CONVEXITY_TOL, min_eig


def estimate_L3(problem, x0: np.ndarray) -> float:
    """Estimate Hessian Lipschitz constant via finite differences."""
    H0 = np.asarray(problem.ihess(x0), dtype=float)
    rng = np.random.default_rng(0)
    L3 = 0.0
    for _ in range(L3_N_DIRS):
        v = rng.normal(size=len(x0))
        v /= np.linalg.norm(v)
        H1 = np.asarray(problem.ihess(x0 + L3_EPS * v), dtype=float)
        L3 = max(L3, np.linalg.norm(H1 - H0, "fro") / L3_EPS)
    return max(L3, 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "6_usefull" / "appendix_a_cutest_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N Appendix A problems.",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        help="Optional explicit problem name subset.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum iterations for each solver.",
    )
    parser.add_argument(
        "--tol-grad",
        type=float,
        default=1e-6,
        help="Gradient-norm stopping tolerance.",
    )
    parser.add_argument(
        "--acrn-l3",
        type=float,
        default=1.0,
        help="Global L3/sigma surrogate used by ACRN.",
    )
    parser.add_argument(
        "--arc-step-method",
        choices=("cauchy", "secular"),
        default="secular",
        help="ARC subproblem step computation mode.",
    )
    return parser.parse_args()


def iter_selected_problems(
    names: Optional[Sequence[str]], limit: Optional[int]
) -> List[Dict[str, object]]:
    problems = APPENDIX_A_PROBLEMS
    if names:
        selected = {name.upper() for name in names}
        problems = [item for item in problems if str(item["name"]).upper() in selected]
    if limit is not None:
        problems = problems[:limit]
    return problems


def candidate_sif_params(
    name: str,
    target_n: int,
    properties: Optional[Dict[str, object]] = None,
) -> Iterable[Optional[Dict[str, int]]]:
    yielded = set()
    candidates: List[Optional[Dict[str, int]]] = [
        None,
        SPECIAL_SIF_PARAMS.get(name),
        {"N": int(target_n)},
    ]
    for params in candidates:
        key = None if params is None else tuple(sorted(params.items()))
        if key in yielded:
            continue
        yielded.add(key)
        yield params


def import_problem(name: str, target_n: int, properties: Optional[Dict[str, object]] = None):
    errors: List[str] = []
    for params in candidate_sif_params(name, target_n, properties=properties):
        try:
            if params is None:
                problem = pycutest.import_problem(name)
            else:
                problem = pycutest.import_problem(name, sifParams=params)
        except Exception as exc:  # pragma: no cover - CUTEst-side variability
            label = "default" if params is None else str(params)
            errors.append(f"{label}: {exc}")
            continue
        actual_n = int(problem.n)
        if actual_n == int(target_n):
            return problem, params, None
        if params is None and actual_n > 0:
            # A default import with the wrong dimension is not useful here.
            continue
        return None, params, f"dimension mismatch (wanted n={target_n}, got n={actual_n})"
    if not errors:
        return None, None, "unable to import problem"
    return None, None, " | ".join(errors)


def make_solver(
    solver_name: str,
    adapter: ProblemAdapter,
    max_iter: int,
    tol_grad: float,
    acrn_l3: float,
    arc_step_method: str,
):
    if solver_name == "NCR":
        signature = inspect.signature(ncr_module.CubicNewton)
        if "options" in signature.parameters:
            return ncr_module.CubicNewton(
                adapter.f,
                adapter.grad,
                adapter.hess,
                options=ncr_module.CRNOptions(max_iter=max_iter, tol_grad=tol_grad),
            )
        kwargs = {}
        for key, value in {
            "tol_grad": tol_grad,
            "max_iter": max_iter,
            "verbose": False,
        }.items():
            if key in signature.parameters:
                kwargs[key] = value
        return ncr_module.CubicNewton(adapter.f, adapter.grad, adapter.hess, **kwargs)
    if solver_name == "ARC":
        signature = inspect.signature(arc_module.AdaptiveCubicNewton)
        if "params" in signature.parameters:
            kwargs = {
                "params": arc_module.ARCParams(
                    max_iter=max_iter,
                    tol_grad=tol_grad,
                ),
            }
            if "step_method" in signature.parameters:
                kwargs["step_method"] = arc_step_method
            return arc_module.AdaptiveCubicNewton(
                adapter.f,
                adapter.grad,
                adapter.hess,
                **kwargs,
            )
        kwargs = {}
        for key, value in {
            "tol_grad": tol_grad,
            "max_iter": max_iter,
            "step_method": arc_step_method,
            "verbose": False,
        }.items():
            if key in signature.parameters:
                kwargs[key] = value
        return arc_module.AdaptiveCubicNewton(adapter.f, adapter.grad, adapter.hess, **kwargs)
    if solver_name == "ACRN":
        signature = inspect.signature(acrn_module.AcceleratedCubicNewton)
        kwargs = {}
        if "L3" in signature.parameters:
            kwargs["L3"] = acrn_l3
        elif "M0" in signature.parameters:
            kwargs["M0"] = acrn_l3
        for key, value in {
            "sigma": acrn_l3,
            "N": 12.0 * acrn_l3,
            "tol_grad": tol_grad,
            "max_iter": max_iter,
            "verbose": False,
        }.items():
            if key in signature.parameters:
                kwargs[key] = value
        return acrn_module.AcceleratedCubicNewton(
            adapter.f,
            adapter.grad,
            adapter.hess,
            **kwargs,
        )
    raise ValueError(f"Unknown solver: {solver_name}")


def extract_iterations(name: str, solver, run_info: Optional[Dict[str, object]]) -> int:
    if name == "ACRN":
        if run_info is None:
            return 0
        return int(run_info.get("iterations", 0))
    return len(getattr(solver, "log", []))


def extract_status(name: str, solver, run_info: Optional[Dict[str, object]], final_grad_norm: float, tol_grad: float) -> str:
    if final_grad_norm <= tol_grad:
        return "converged"
    if name == "ACRN":
        return "max_iter"
    return str(getattr(solver, "termination_reason", "unknown"))


def run_solver(
    solver_name: str,
    solver,
    adapter: ProblemAdapter,
    x0: np.ndarray,
    tol_grad: float,
) -> Dict[str, object]:
    start = time.perf_counter()
    run_info: Optional[Dict[str, object]] = None
    try:
        if solver_name == "ACRN":
            x_star, run_info = solver.run(x0)
        else:
            x_star = solver.run(x0)
        elapsed = time.perf_counter() - start
        final_f = float(adapter.f(x_star))
        final_grad_norm = float(np.linalg.norm(adapter.grad(x_star)))
        status = extract_status(solver_name, solver, run_info, final_grad_norm, tol_grad)
        return {
            "status": status,
            "message": "",
            "runtime_sec": elapsed,
            "iterations": extract_iterations(solver_name, solver, run_info),
            "final_f": final_f,
            "final_grad_norm": final_grad_norm,
            "x_norm": float(np.linalg.norm(x_star)),
        }
    except Exception as exc:  # pragma: no cover - solver failures depend on CUTEst instance
        elapsed = time.perf_counter() - start
        return {
            "status": "error",
            "message": str(exc),
            "runtime_sec": elapsed,
            "iterations": extract_iterations(solver_name, solver, run_info),
            "final_f": math.nan,
            "final_grad_norm": math.nan,
            "x_norm": math.nan,
        }


def benchmark_problem(
    spec: Dict[str, object],
    max_iter: int,
    tol_grad: float,
    acrn_l3: float,
    arc_step_method: str,
) -> List[Dict[str, object]]:
    name = str(spec["name"])
    target_n = int(spec["n"])
    try:
        properties = pycutest.problem_properties(name)
    except Exception as exc:  # pragma: no cover - CUTEst-side variability
        properties = {"constraints": "unknown", "objective": "unknown"}
        property_message = str(exc)
    else:
        property_message = ""
    problem, params, import_message = import_problem(name, target_n, properties=properties)
    rows: List[Dict[str, object]] = []

    if problem is None:
        for solver_name in ("NCR", "ARC", "ACRN"):
            rows.append(
                {
                    "problem": name,
                    "target_n": target_n,
                    "actual_n": "",
                    "solver": solver_name,
                    "status": "skipped",
                    "message": import_message or property_message or "import failed",
                    "sif_params": params or "",
                    "constraints": properties.get("constraints", ""),
                    "objective": properties.get("objective", ""),
                    "is_convex": "",
                    "min_eig_x0": "",
                    "l3_used": "",
                    "iterations": "",
                    "f_calls": "",
                    "g_calls": "",
                    "h_calls": "",
                    "final_f": "",
                    "final_grad_norm": "",
                    "runtime_sec": "",
                    "x0_norm": "",
                    "x_norm": "",
                }
            )
        return rows

    actual_n = int(problem.n)
    if getattr(problem, "m", 0):
        for solver_name in ("NCR", "ARC", "ACRN"):
            rows.append(
                {
                    "problem": name,
                    "target_n": target_n,
                    "actual_n": actual_n,
                    "solver": solver_name,
                    "status": "skipped",
                    "message": f"constrained problem (m={int(problem.m)})",
                    "sif_params": params or "",
                    "constraints": properties.get("constraints", ""),
                    "objective": properties.get("objective", ""),
                    "is_convex": "",
                    "min_eig_x0": "",
                    "l3_used": "",
                    "iterations": "",
                    "f_calls": "",
                    "g_calls": "",
                    "h_calls": "",
                    "final_f": "",
                    "final_grad_norm": "",
                    "runtime_sec": "",
                    "x0_norm": float(np.linalg.norm(problem.x0)),
                    "x_norm": "",
                }
            )
        return rows

    x0 = np.asarray(problem.x0, dtype=float)
    is_convex, min_eig_x0 = check_convexity(problem, properties)
    l3_used: float = math.nan
    if is_convex:
        l3_used = estimate_L3(problem, x0)

    for solver_name in ("NCR", "ARC", "ACRN"):
        if solver_name == "ACRN" and not is_convex:
            rows.append(
                {
                    "problem": name,
                    "target_n": target_n,
                    "actual_n": actual_n,
                    "solver": "ACRN",
                    "status": "skipped_nonconvex",
                    "message": f"min_eig(H0)={min_eig_x0:.3e}",
                    "sif_params": params or "",
                    "constraints": properties.get("constraints", ""),
                    "objective": properties.get("objective", ""),
                    "is_convex": is_convex,
                    "min_eig_x0": min_eig_x0,
                    "l3_used": "",
                    "iterations": "",
                    "f_calls": "",
                    "g_calls": "",
                    "h_calls": "",
                    "final_f": "",
                    "final_grad_norm": "",
                    "runtime_sec": "",
                    "x0_norm": float(np.linalg.norm(x0)),
                    "x_norm": "",
                }
            )
            continue

        effective_l3 = l3_used if solver_name == "ACRN" else acrn_l3
        adapter = ProblemAdapter(problem)
        solver = make_solver(
            solver_name,
            adapter,
            max_iter=max_iter,
            tol_grad=tol_grad,
            acrn_l3=effective_l3,
            arc_step_method=arc_step_method,
        )
        result = run_solver(solver_name, solver, adapter, x0.copy(), tol_grad)
        rows.append(
            {
                "problem": name,
                "target_n": target_n,
                "actual_n": actual_n,
                "solver": solver_name,
                "status": result["status"],
                "message": result["message"],
                "sif_params": params or "",
                "constraints": properties.get("constraints", ""),
                "objective": properties.get("objective", ""),
                "is_convex": is_convex,
                "min_eig_x0": min_eig_x0,
                "l3_used": l3_used if solver_name == "ACRN" else "",
                "iterations": result["iterations"],
                "f_calls": adapter.counter.f_calls,
                "g_calls": adapter.counter.g_calls,
                "h_calls": adapter.counter.h_calls,
                "final_f": result["final_f"],
                "final_grad_norm": result["final_grad_norm"],
                "runtime_sec": result["runtime_sec"],
                "x0_norm": float(np.linalg.norm(x0)),
                "x_norm": result["x_norm"],
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "problem",
        "target_n",
        "actual_n",
        "solver",
        "status",
        "message",
        "sif_params",
        "constraints",
        "objective",
        "is_convex",
        "min_eig_x0",
        "l3_used",
        "iterations",
        "f_calls",
        "g_calls",
        "h_calls",
        "final_f",
        "final_grad_norm",
        "runtime_sec",
        "x0_norm",
        "x_norm",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


_SOLVERS = ["NCR", "ARC", "ACRN"]
_WIDE_FIELDNAMES = ["problem", "n"] + [
    f"{s}_{col}"
    for s in _SOLVERS
    for col in ["iter", "iter_ok", "g", "h", "f", "gnorm", "status"]
]


def write_wide_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write one row per problem with per-solver columns (Appendix A format)."""
    # Group rows by problem name, preserving order.
    from collections import defaultdict
    by_problem: Dict[str, Dict[str, Dict]] = {}
    for row in rows:
        name = str(row["problem"])
        solver = str(row["solver"])
        if name not in by_problem:
            by_problem[name] = {}
        by_problem[name][solver] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_WIDE_FIELDNAMES)
        writer.writeheader()
        for name, solver_rows in by_problem.items():
            # Use any available row for problem-level fields.
            any_row = next(iter(solver_rows.values()))
            wide: Dict[str, object] = {
                "problem": name,
                "n": any_row.get("actual_n") or any_row.get("target_n", ""),
            }
            for s in _SOLVERS:
                r = solver_rows.get(s, {})
                iters = r.get("iterations", "")
                rejected = r.get("rejected_steps", "")
                if iters != "" and rejected != "":
                    iter_ok = int(iters) - int(rejected)
                elif iters != "":
                    iter_ok = iters  # ACRN has no rejections
                else:
                    iter_ok = ""
                wide[f"{s}_iter"]   = iters
                wide[f"{s}_iter_ok"] = iter_ok
                wide[f"{s}_g"]      = r.get("g_calls", "")
                wide[f"{s}_h"]      = r.get("h_calls", "")
                wide[f"{s}_f"]      = r.get("final_f", "")
                wide[f"{s}_gnorm"]  = r.get("final_grad_norm", "")
                wide[f"{s}_status"] = r.get("status", "")
            writer.writerow(wide)


def main() -> None:
    args = parse_args()
    selected = iter_selected_problems(args.problems, args.limit)
    rows: List[Dict[str, object]] = []

    print(
        f"Running {len(selected)} CUTEst problems with max_iter={args.max_iter}, "
        f"tol_grad={args.tol_grad:.1e}, ACRN L3={args.acrn_l3:.2e}"
    )

    for index, spec in enumerate(selected, start=1):
        name = str(spec["name"])
        target_n = int(spec["n"])
        print(f"[{index:03d}/{len(selected):03d}] {name} (target n={target_n})")
        problem_rows = benchmark_problem(
            spec,
            max_iter=args.max_iter,
            tol_grad=args.tol_grad,
            acrn_l3=args.acrn_l3,
            arc_step_method=args.arc_step_method,
        )
        rows.extend(problem_rows)
        for row in problem_rows:
            line = (
                f"  {row['solver']:<4} status={row['status']:<9} "
                f"iter={row['iterations']} f={row['final_f']}"
            )
            if row["message"]:
                line += f" | {row['message']}"
            print(line)

    output_path = Path(args.output)
    write_csv(rows, output_path)
    print(f"Saved results to {output_path}")

    wide_path = output_path.with_stem(output_path.stem + "_wide")
    write_wide_csv(rows, wide_path)
    print(f"Saved wide results to {wide_path}")


if __name__ == "__main__":
    main()
