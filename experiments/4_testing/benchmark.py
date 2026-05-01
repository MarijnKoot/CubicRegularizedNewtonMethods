"""
Benchmark: PureNewton vs NCR vs ARC vs ACRN on standard test functions.

Outputs
-------
- experiments/4_testing/results.csv
- LaTeX table printed to stdout

Usage
-----
    python experiments/4_testing/benchmark.py
"""

from __future__ import annotations

import csv
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

SOLVER_TIMEOUT_S = 60   # wall-clock seconds per solver call


class _SolverTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SolverTimeout("solver timed out")

import numpy as np
from numpy.linalg import norm

# ── path setup ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]   # /home/marijn/Thesis
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utilities import estimate_L3
from methods.pure_newton import PureNewton, PureNewtonOptions
from methods.NCR import CubicNewton, CRNOptions
from methods.ARC import AdaptiveCubicNewton, ARCParams
from methods.ACRN import AcceleratedCubicNewton

from test_functions.quadratic import quadratic
from test_functions.cubic_norm import cubic_norm
from test_functions.quartic_convex import quartic_convex
from test_functions.rosenbrock import rosenbrock
from test_functions.dixon_price import dixon_price
from test_functions.logsumexp import logsumexp
from test_functions.rastrigin import rastrigin

# ── shared tolerances ─────────────────────────────────────────────────────────

TOL_GRAD = 1e-6
MAX_ITER = 500

# ── utilities ─────────────────────────────────────────────────────────────────

def is_convex_at(hess_fn, x, tol: float = -1e-6) -> bool:
    H = np.asarray(hess_fn(x), dtype=float)
    H = 0.5 * (H + H.T)
    return float(np.linalg.eigvalsh(H).min()) >= tol




# ── test-function factories ───────────────────────────────────────────────────
# Each factory(n) -> (f, grad, hess, L3_or_None, x0, x0_desc)
# L3_or_None = None means it will be estimated numerically.
# x0_desc is a LaTeX string describing the starting point.

def make_quadratic(n: int):
    rng = np.random.default_rng(42)
    M = rng.normal(size=(n, n))
    A = M.T @ M + np.eye(n)          # symmetric PD, well-conditioned
    b = np.zeros(n)                  # x* = 0, f* = 0
    f, grad, hess, L3_exact = quadratic(A, b)
    x0 = np.ones(n) * 5.0
    return f, grad, hess, L3_exact, x0, r"$5 \cdot \mathbf{1}_n$"


def make_cubic_norm(n: int):
    f, grad, hess, L3_exact = cubic_norm(n)
    x0 = np.ones(n) * 2.0
    return f, grad, hess, L3_exact, x0, r"$2 \cdot \mathbf{1}_n$"


def make_quartic_convex(n: int):
    f, grad, hess = quartic_convex(n)
    x0 = np.ones(n) * 2.0
    return f, grad, hess, None, x0, r"$2 \cdot \mathbf{1}_n$"   # L3 estimated below


def make_rosenbrock(n: int):
    f, grad, hess = rosenbrock(n)
    x0 = np.full(n, -1.0)            # classic hard start
    return f, grad, hess, None, x0, r"$-\mathbf{1}_n$"


def make_dixon_price(n: int):
    f, grad, hess = dixon_price(n)
    x0 = np.ones(n) * 2.0
    return f, grad, hess, None, x0, r"$2 \cdot \mathbf{1}_n$"


def make_rastrigin(n: int):
    f, grad, hess = rastrigin(n)
    x0 = np.ones(n) * 2.0
    return f, grad, hess, None, x0, r"$2 \cdot \mathbf{1}_n$"


def make_logsumexp(n: int):
    rng = np.random.default_rng(42)
    m = 2 * n                         # 2× over-determined system
    a = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    mu = 1.0
    f, grad, hess, _ = logsumexp(a, b, mu)
    x0 = np.zeros(n)
    return f, grad, hess, None, x0, r"$\mathbf{0}_n$"


# ── problem registry ──────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name":          "quadratic",
        "dims":          [10, 50, 100],
        "factory":       make_quadratic,
        "convex":        True,
        "skip_acrn":     False,
        "sigma0_override": 0.0,   # L3=0 exactly; sigma=0 → pure Newton step → 1-iter convergence
    },
    {
        "name":      "cubic norm",
        "dims":      [2, 10, 50],
        "factory":   make_cubic_norm,
        "convex":    True,
        "skip_acrn": False,
    },
    {
        "name":      "quartic convex",
        "dims":      [2, 10, 50],
        "factory":   make_quartic_convex,
        "convex":    True,
        "skip_acrn": False,
    },
    {
        "name":      "logsumexp",
        "dims":      [10, 50, 100],
        "factory":   make_logsumexp,
        "convex":    True,
        "skip_acrn": False,
    },
    {
        "name":          "rastrigin",
        "dims":          [2, 10, 50],
        "factory":       make_rastrigin,
        "convex":        False,
        "skip_acrn":     True,    # highly non-convex with many local minima
    },
    {
        "name":      "rosenbrock",
        "dims":      [2, 10, 50],
        "factory":   make_rosenbrock,
        "convex":    False,
        "skip_acrn": True,    # non-convex
    },
    {
        "name":      "Dixon-Price",
        "dims":      [2, 10, 50],
        "factory":   make_dixon_price,
        "convex":    False,
        "skip_acrn": True,    # non-convex at start
    },
]

# ── solver runner ─────────────────────────────────────────────────────────────

def build_solvers(f, grad, hess, L3: float, sigma0: float, skip_acrn: bool):
    """Return list of (label, solver_object) pairs to benchmark."""
    sigma0_pos = max(sigma0, 1e-12)    # solvers require strictly positive sigma0
    solvers = [
        (
            "PureNewton",
            PureNewton(f, grad, hess, PureNewtonOptions(tol_grad=TOL_GRAD, max_iter=MAX_ITER)),
        ),
        (
            "NCR",
            CubicNewton(f, grad, hess, CRNOptions(sigma0=sigma0_pos, tol_grad=TOL_GRAD, max_iter=MAX_ITER)),
        ),
        (
            "ARC",
            AdaptiveCubicNewton(
                f, grad, hess,
                ARCParams(sigma0=sigma0_pos, tol_grad=TOL_GRAD, max_iter=MAX_ITER),
                step_method="secular",
            ),
        ),
    ]
    if not skip_acrn:
        acrn_L3    = max(L3, 1e-12)    # guard against exactly-zero L3 (e.g. quadratic)
        acrn_sigma = max(sigma0, 1e-12)
        solvers.append(
            (
                "ACRN",
                AcceleratedCubicNewton(
                    f, grad, hess,
                    L3=acrn_L3,
                    sigma=acrn_sigma,
                    tol_grad=TOL_GRAD,
                    max_iter=MAX_ITER,
                    adaptive_sigma=False,
                ),
            )
        )
    return solvers


def run_one(solver, x0: np.ndarray):
    """Run a solver and return (x_star, elapsed_s). Raises _SolverTimeout if too slow."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(SOLVER_TIMEOUT_S)
    try:
        t0 = time.perf_counter()
        result = solver.run(x0.copy())
        elapsed = time.perf_counter() - t0
    finally:
        signal.alarm(0)   # cancel alarm whether we succeeded or not
    x_star = result[0] if isinstance(result, tuple) else result
    return np.asarray(x_star, dtype=float), elapsed


def count_rejected(solver_name: str, log: list) -> int:
    if solver_name == "ACRN":
        return int(sum(e.get("rejected_trials", 0) for e in log))
    return int(sum(not e.get("accepted", True) for e in log))


def l3_hat(solver_name: str, L3: float) -> Optional[float]:
    """Report the (estimated) Hessian Lipschitz constant for all cubic methods."""
    if solver_name == "PureNewton":
        return None
    return float(L3)


# ── main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark() -> list[dict]:
    rows = []

    for tc in TEST_CASES:
        for n in tc["dims"]:
            print(f"  {tc['name']} n={n} ...", end="", flush=True)
            f, grad, hess, L3_exact, x0, x0_desc = tc["factory"](n)

            # Resolve L3
            L3 = L3_exact if L3_exact is not None else estimate_L3(hess, x0)

            # Choose sigma0; allow per-problem override (e.g. quadratic uses 0)
            if "sigma0_override" in tc:
                sigma0 = float(tc["sigma0_override"])
            else:
                sigma0 = max(L3 / 2.0, 0.5)

            # Decide whether to run ACRN (skip if non-convex at x0)
            skip_acrn = tc["skip_acrn"] or not is_convex_at(hess, x0)

            solvers = build_solvers(f, grad, hess, L3, sigma0, skip_acrn)

            for label, solver in solvers:
                row = {
                    "function":    tc["name"],
                    "convex":      tc["convex"],
                    "dim":         n,
                    "solver":      label,
                    "x0_desc":     x0_desc,
                    "x0_norm":     float(norm(x0)),
                    "f_x0":        float(f(x0)),
                    "iter":        None,
                    "rejected":    None,
                    "time_s":      None,
                    "f_star":      None,
                    "grad_norm":   None,
                    "l3_hat":      l3_hat(label, L3),
                    "termination": None,
                    "status":      "ok",
                }
                try:
                    x_star, elapsed = run_one(solver, x0)

                    row["iter"]        = len(solver.log)
                    row["rejected"]    = count_rejected(label, solver.log)
                    row["time_s"]      = elapsed
                    row["f_star"]      = float(f(x_star))
                    row["grad_norm"]   = float(norm(np.asarray(grad(x_star), dtype=float)))
                    row["termination"] = solver.termination_reason

                except _SolverTimeout:
                    row["status"] = f"TIMEOUT (>{SOLVER_TIMEOUT_S}s)"
                    print(f"\n    [{label}] TIMEOUT", end="")
                except Exception as exc:
                    row["status"]      = f"FAILED: {exc}"
                    print(f"\n    [{label}] ERROR: {exc}")
                    traceback.print_exc()

                rows.append(row)
                print(f" {label}", end="", flush=True)

            print()   # newline after each (function, dim) group

    return rows


# ── output: CSV ───────────────────────────────────────────────────────────────

CSV_COLS = [
    "function", "convex", "dim", "solver",
    "x0_desc", "x0_norm", "f_x0",
    "iter", "rejected", "time_s",
    "f_star", "grad_norm",
    "l3_hat", "termination", "status",
]

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written to: {path}")


# ── output: LaTeX table ───────────────────────────────────────────────────────

def _fmt(val, fmt=".3e") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    return format(val, fmt)


def _solver_table(rows: list[dict], convex: bool) -> None:
    """Print one LaTeX results table, filtered to convex or non-convex problems."""
    subset = [r for r in rows if r["status"] == "ok" and r["convex"] == convex]
    if not subset:
        return

    caption = (
        "Solver comparison on convex test functions."
        if convex else
        "Solver comparison on non-convex test functions."
    )
    label = "tab:solver-convex" if convex else "tab:solver-nonconvex"

    header = (
        r"\begin{table}[H]"                                                         "\n"
        r"\centering"                                                               "\n"
        r"\small"                                                                   "\n"
        f"\\caption{{{caption}}}"                                                   "\n"
        f"\\label{{{label}}}"                                                       "\n"
        r"\begin{tabular}{llrl|rrrrrr}"                                             "\n"
        r"\toprule"                                                                 "\n"
        r"Function & Solver & $n$ & $x_0$ & Iter & Rej & Time (s) "
        r"& $f(x^*)$ & $\|\nabla f(x^*)\|$ "
        r"& $\hat{L}_3$ \\"                                                         "\n"
        r"\midrule"
    )
    print(header)

    current_fn = None
    for row in subset:
        if row["function"] != current_fn:
            if current_fn is not None:
                print(r"\midrule")
            current_fn = row["function"]

        l3_str = _fmt(row["l3_hat"], ".2e") if row["l3_hat"] is not None else "--"
        line = (
            f"{row['function']} & {row['solver']} & ${row['dim']}$ & "
            f"{row['x0_desc']} & "
            f"${row['iter']}$ & ${row['rejected']}$ & "
            f"${_fmt(row['time_s'], '.3f')}$ & "
            f"${_fmt(row['f_star'])}$ & "
            f"${_fmt(row['grad_norm'])}$ & "
            f"${l3_str}$ \\\\"
        )
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_latex(rows: list[dict]) -> None:
    """Print convex table, then non-convex table."""
    _solver_table(rows, convex=True)
    print()
    _solver_table(rows, convex=False)


def print_latex_parameters() -> None:
    """Print a table of test-function definitions, parameters, and known constants."""
    rows = [
        # (name, convex, formula, parameters, L3, x_star, f_star)
        (
            "Quadratic",
            "yes",
            r"$\frac{1}{2}x^\top Ax - b^\top x$",
            r"$A = M^\top M + I_n$, $M \sim \mathcal{N}(0,1)$, $b = \mathbf{0}_n$, seed 42",
            r"$0$",
            r"$\mathbf{0}_n$",
            r"$0$",
        ),
        (
            "Cubic norm",
            "yes",
            r"$\frac{1}{3}\|x\|^3$",
            r"---",
            r"$2$",
            r"$\mathbf{0}_n$",
            r"$0$",
        ),
        (
            "Quartic convex",
            "yes",
            r"$\frac{1}{2}\|x\|^2 + \frac{1}{4}\|x\|^4$",
            r"---",
            r"$\hat{L}_3$",
            r"$\mathbf{0}_n$",
            r"$0$",
        ),
        (
            "Log-sum-exp",
            "yes",
            r"$\mu\log\!\sum_{i=1}^m \exp\!\bigl(\tfrac{a_i^\top x - b_i}{\mu}\bigr)$",
            r"$a \sim \mathcal{N}(0,1)$, $m \times n$; $b \sim \mathcal{N}(0,1)$; $m = 2n$, $\mu = 1$, seed 42",
            r"$\hat{L}_3$",
            r"---",
            r"---",
        ),
        (
            "Rosenbrock",
            "no",
            r"$\sum_{i=0}^{n-2}\bigl[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2\bigr]$",
            r"---",
            r"$\hat{L}_3$",
            r"$\mathbf{1}_n$",
            r"$0$",
        ),
        (
            "Dixon-Price",
            "no",
            r"$(x_0-1)^2 + \sum_{i=1}^{n-1}(i+1)(2x_i^2 - x_{i-1})^2$",
            r"---",
            r"$\hat{L}_3$",
            r"$x^*_i = 2^{-(2^i-2)/2^i}$",
            r"$0$",
        ),
        (
            "Rastrigin",
            "no",
            r"$An + \sum_{i=1}^n\bigl[x_i^2 - A\cos(2\pi x_i)\bigr]$",
            r"$A = 10$",
            r"$\hat{L}_3$",
            r"$\mathbf{0}_n$",
            r"$0$",
        ),
    ]

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\caption{Test function definitions, parameters, and known constants.}")
    print(r"\label{tab:test-functions}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{llp{4.5cm}p{5.5cm}lll}")
    print(r"\toprule")
    print(r"Function & Convex & Definition & Parameters & $L_3$ & $x^*$ & $f^*$ \\")
    print(r"\midrule")
    for name, convex, formula, params, L3, x_star, f_star in rows:
        print(f"{name} & {convex} & {formula} & {params} & {L3} & {x_star} & {f_star} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


def print_latex_initial_points(rows: list[dict]) -> None:
    """Print a table of initial points (one row per (function, n) pair)."""
    seen: set = set()
    entries = []
    for r in rows:
        key = (r["function"], r["dim"])
        if key not in seen:
            seen.add(key)
            entries.append((r["function"], r["convex"], r["dim"], r["x0_desc"], r["x0_norm"], r["f_x0"]))

    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{Initial points used in the benchmark.}")
    print(r"\label{tab:initial-points}")
    print(r"\begin{tabular}{llr|lrr}")
    print(r"\toprule")
    print(r"Function & Convex & $n$ & $x_0$ & $\|x_0\|$ & $f(x_0)$ \\")
    print(r"\midrule")

    current_fn = None
    for fn, convex, n, x0_desc, x0_norm, f_x0 in entries:
        if fn != current_fn:
            if current_fn is not None:
                print(r"\midrule")
            current_fn = fn
        convex_str = "yes" if convex else "no"
        print(f"{fn} & {convex_str} & ${n}$ & {x0_desc} & ${x0_norm:.3e}$ & ${f_x0:.3e}$ \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── plain-text summary table ──────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        print("No successful runs.")
        return

    col_w = [14, 12, 5, 6, 5, 9, 11, 11, 10, 12]
    headers = [
        "Function", "Solver", "n", "Iter", "Rej", "Time(s)",
        "f(x*)", "||grad||", "L3_hat", "stop",
    ]

    def row_str(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    sep = "-" * sum(col_w + [2 * (len(col_w) - 1)])
    print("\n" + sep)
    print(row_str(headers))
    print(sep)

    current_fn = None
    for r in ok:
        if r["function"] != current_fn:
            if current_fn is not None:
                print(sep)
            current_fn = r["function"]
        l3_str = f"{r['l3_hat']:.2e}" if r["l3_hat"] is not None else "--"
        vals = [
            r["function"],
            r["solver"],
            r["dim"],
            r["iter"],
            r["rejected"],
            f"{r['time_s']:.4f}",
            f"{r['f_star']:.3e}",
            f"{r['grad_norm']:.3e}",
            l3_str,
            r["termination"],
        ]
        print(row_str(vals))
    print(sep)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("Running benchmark ...\n")
    rows = run_benchmark()

    out_csv = ROOT / "results" / "4_testing" / "results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)

    print("\n" + "=" * 80)
    print("PLAIN-TEXT SUMMARY")
    print("=" * 80)
    print_summary(rows)

    print("\n" + "=" * 80)
    print("LATEX — TEST FUNCTION PARAMETERS")
    print("=" * 80 + "\n")
    print_latex_parameters()

    print("\n" + "=" * 80)
    print("LATEX — INITIAL POINTS")
    print("=" * 80 + "\n")
    print_latex_initial_points(rows)

    print("\n" + "=" * 80)
    print("LATEX — CONVEX RESULTS")
    print("=" * 80 + "\n")
    _solver_table(rows, convex=True)

    print("\n" + "=" * 80)
    print("LATEX — NON-CONVEX RESULTS")
    print("=" * 80 + "\n")
    _solver_table(rows, convex=False)


if __name__ == "__main__":
    main()
