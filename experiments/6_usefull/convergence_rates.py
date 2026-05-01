"""
Convergence rate comparison: Newton, NCR, ARC, ACRN on a suite of test functions.

For each (problem, solver) pair records the per-iteration gradient-norm trajectory
||∇f(x_k)|| and function-value excess f(x_k) - f*.  Plots and raw CSV are written
to results/.

Usage (from repo root, inside venv):
    python experiments/convergence_rates.py

Outputs:
    results/convergence_rates.csv
    results/figures/convergence_<problem>.pdf
    results/figures/performance_profile.pdf
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from methods.NCR        import CubicNewton, CRNOptions
from methods.ARC        import AdaptiveCubicNewton, ARCParams
from methods.ACRN       import AcceleratedCubicNewton
from methods.pure_newton import PureNewton, PureNewtonOptions
from test_functions.quadratic      import quadratic
from test_functions.cubic_norm     import cubic_norm
from test_functions.cubic_poly     import cubic_poly
from test_functions.quartic_convex import quartic_convex
from test_functions.logsumexp      import logsumexp
from test_functions.rosenbrock     import rosenbrock
from utilities import estimate_L3, is_convex_at

# ── settings ───────────────────────────────────────────────────────────────────
SEED     = 42
TOL_GRAD = 1e-6
MAX_ITER = 300

RESULTS_DIR = ROOT / "results" / "6_usefull"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_COLORS  = {"Newton": "tab:gray", "NCR": "tab:blue",
                  "ARC": "tab:orange",  "ACRN": "tab:green"}
SOLVER_LS      = {"Newton": "-",  "NCR": "-",  "ARC": "-",  "ACRN": "-"}
SOLVER_MARKERS = {"Newton": "s",  "NCR": "o",  "ARC": "^",  "ACRN": "D"}

# Theoretical rate references shown on f(x_k)-f* plot.
# Entry: solver -> (exponent p, linestyle, label)  →  C·k^{-p} reference line.
THEORY_REFS = {
    "NCR":  (2, "--", r"$O(k^{-2})$"),
    "ARC":  (2, ":",  r"$O(k^{-2})$"),
    "ACRN": (3, "-.", r"$O(k^{-3})$"),
}

rng = np.random.default_rng(SEED)


# ── problem constructors ────────────────────────────────────────────────────────

def _make_spd(n: int, cond: float) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs  = np.linspace(1.0, float(cond), n)
    return Q @ np.diag(eigs) @ Q.T


def build_problem_suite() -> List[Dict]:
    """
    Each entry is a dict:
        name    : str   — filesystem-safe identifier
        label   : str   — human-readable label for plot titles
        f, grad, hess : callables
        L3      : float or None  (Hessian Lipschitz constant; None = estimate)
        f_star  : float or None  (true minimum; None = use observed minimum)
        x0      : ndarray
        convex  : bool
    """
    suite = []

    # ── Quadratic (L3 = 0 exactly; Newton/NCR/ARC should converge in ~1 step) ─
    for n, cond in [(5, 100), (20, 1_000)]:
        A = _make_spd(n, cond)
        b = rng.normal(size=n)
        f, grad, hess, L3 = quadratic(A, b)
        x0 = rng.normal(size=n) * 3.0
        # f* = -0.5 b^T A^{-1} b  (known)
        x_star_quad = np.linalg.solve(A, b)
        f_star = float(f(x_star_quad))
        suite.append(dict(
            name=f"quadratic_n{n}_cond{int(cond)}",
            label=f"Quadratic (n={n}, κ={cond:.0e})",
            f=f, grad=grad, hess=hess,
            L3=L3, f_star=f_star, x0=x0, convex=True,
        ))

    # ── Cubic norm  (L3 = 2 exactly; f* = 0 at origin) ────────────────────────
    for n in [2, 10]:
        f, grad, hess, L3 = cubic_norm(n=n)
        x0 = rng.normal(size=n) * 2.0
        suite.append(dict(
            name=f"cubic_norm_n{n}",
            label=f"Cubic norm (n={n})",
            f=f, grad=grad, hess=hess,
            L3=L3, f_star=0.0, x0=x0, convex=True,
        ))

    # ── Cubic poly (L3 = 2γ; f* = 0 at origin) ────────────────────────────────
    Q2 = np.array([[2.0, 0.5], [0.5, 1.0]])
    f, grad, hess, L3 = cubic_poly(Q2, gamma=1.0)
    x0 = rng.normal(size=2) * 2.0
    suite.append(dict(
        name="cubic_poly_n2",
        label="Cubic poly (n=2, γ=1)",
        f=f, grad=grad, hess=hess,
        L3=L3, f_star=0.0, x0=x0, convex=True,
    ))

    # ── Quartic convex (f* = 0 at origin) ─────────────────────────────────────
    for n in [2, 10]:
        f, grad, hess = quartic_convex(n=n)
        x0 = rng.normal(size=n) * 2.0
        L3 = estimate_L3(hess, x0)
        suite.append(dict(
            name=f"quartic_convex_n{n}",
            label=f"Quartic convex (n={n})",
            f=f, grad=grad, hess=hess,
            L3=L3, f_star=0.0, x0=x0, convex=True,
        ))

    # ── LogSumExp (f* unknown analytically → use observed minimum) ────────────
    for n, mu in [(5, 0.5), (20, 0.2)]:
        m     = 4 * n
        a_mat = rng.normal(size=(m, n))
        b_vec = rng.normal(size=m)
        f, grad, hess, _ = logsumexp(a_mat, b_vec, mu)
        x0 = np.zeros(n)
        L3 = estimate_L3(hess, x0)
        suite.append(dict(
            name=f"logsumexp_n{n}_mu{mu}",
            label=f"LogSumExp (n={n}, μ={mu})",
            f=f, grad=grad, hess=hess,
            L3=L3, f_star=None, x0=x0, convex=True,
        ))

    # ── Rosenbrock (nonconvex; f* = 0 at all-ones; ACRN skipped) ──────────────
    for N in [2, 5]:
        f, grad, hess = rosenbrock(N)
        x0 = np.zeros(N)
        suite.append(dict(
            name=f"rosenbrock_n{N}",
            label=f"Rosenbrock (n={N})",
            f=f, grad=grad, hess=hess,
            L3=None, f_star=0.0, x0=x0, convex=False,
        ))

    return suite


# ── solver factory ─────────────────────────────────────────────────────────────

def make_solver(sname: str, f, grad, hess, L3: Optional[float], x0: np.ndarray):
    if sname == "Newton":
        opts = PureNewtonOptions(tol_grad=TOL_GRAD, max_iter=MAX_ITER)
        return PureNewton(f, grad, hess, options=opts)

    if sname == "NCR":
        sigma0 = max(2.0 * L3, 1e-6) if (L3 is not None and L3 > 0) else 0.5
        return CubicNewton(f, grad, hess,
            options=CRNOptions(sigma0=sigma0, tol_grad=TOL_GRAD, max_iter=MAX_ITER))

    if sname == "ARC":
        return AdaptiveCubicNewton(f, grad, hess,
            params=ARCParams(tol_grad=TOL_GRAD, max_iter=MAX_ITER),
            step_method="secular")

    if sname == "ACRN":
        L3_eff = L3 if (L3 is not None and L3 > 0) else estimate_L3(hess, x0)
        return AcceleratedCubicNewton(f, grad, hess,
            L3=max(L3_eff, 1e-6), tol_grad=TOL_GRAD, max_iter=MAX_ITER,
            adaptive_sigma=True)

    raise ValueError(f"Unknown solver: {sname}")


# ── trajectory extraction ──────────────────────────────────────────────────────

def extract_trajectory(
    sname: str, solver, grad, f, run_result
) -> Tuple[List[float], List[float]]:
    """
    Return (grad_norms, f_vals) evaluated at the actual iterates x_k.

    For NCR / ARC / Newton, log[k]["grad_norm"] is already ||∇f(x_k)|| computed
    before the step, and log[k]["f"] is f(x_k).

    For ACRN, log[k]["grad_norm"] is ||∇f(y_k)|| (at the interpolated point).
    We instead walk the history.xk sequence and recompute ||∇f(x_k)|| there.
    This requires O(K) extra gradient evaluations but gives a fair comparison.
    """
    if sname == "ACRN" and isinstance(run_result, tuple):
        x_final, info = run_result
        history = info.get("history", [])
        xs = [row.xk for row in history]
        if not xs or not np.allclose(xs[-1], x_final, atol=1e-12):
            xs = xs + [x_final]
        grad_norms = [float(np.linalg.norm(grad(x))) for x in xs]
        f_vals     = [float(f(x)) for x in xs]
    else:
        log = getattr(solver, "log", [])
        grad_norms = [e["grad_norm"] for e in log]
        f_vals     = [e["f"]        for e in log]
    return grad_norms, f_vals


# ── run one problem ────────────────────────────────────────────────────────────

def run_problem(
    prob: Dict, csv_rows: List[Dict]
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Run all applicable solvers on prob; accumulate CSV rows; return trajectories."""
    f, grad, hess = prob["f"], prob["grad"], prob["hess"]
    L3, x0        = prob["L3"], prob["x0"]
    convex        = prob["convex"]

    solver_names = ["Newton", "NCR", "ARC"] + (["ACRN"] if convex else [])
    trajectories: Dict[str, Tuple[List[float], List[float]]] = {}

    for sname in solver_names:
        try:
            solver = make_solver(sname, f, grad, hess, L3, x0)
            t0  = time.perf_counter()
            raw = solver.run(x0.copy())
            elapsed = time.perf_counter() - t0

            x_final         = raw[0] if isinstance(raw, tuple) else raw
            final_grad_norm = float(np.linalg.norm(grad(x_final)))
            final_f         = float(f(x_final))
            iters           = len(getattr(solver, "log", []))
            status          = getattr(solver, "termination_reason", "unknown")
            converged       = final_grad_norm <= TOL_GRAD

            grad_norms, f_vals = extract_trajectory(sname, solver, grad, f, raw)
            trajectories[sname] = (grad_norms, f_vals)

        except Exception as exc:
            final_grad_norm = float("nan")
            final_f         = float("nan")
            iters           = 0
            elapsed         = float("nan")
            status          = f"error: {exc}"
            converged       = False
            grad_norms, f_vals = [], []
            print(f"    [{sname}] ERROR: {exc}")

        csv_rows.append({
            "problem":        prob["name"],
            "solver":         sname,
            "n":              len(x0),
            "convex":         convex,
            "L3":             f"{L3:.4g}" if L3 is not None else "",
            "iterations":     iters,
            "runtime_sec":    f"{elapsed:.6f}" if not np.isnan(elapsed) else "",
            "final_f":        f"{final_f:.6e}" if not np.isnan(final_f) else "",
            "final_grad_norm":f"{final_grad_norm:.6e}" if not np.isnan(final_grad_norm) else "",
            "converged":      converged,
            "status":         status,
        })

    return trajectories


# ── plotting ────────────────────────────────────────────────────────────────────

def plot_problem(prob: Dict, trajectories: Dict, fig_dir: Path) -> None:
    name   = prob["name"]
    label  = prob["label"]
    f_star = prob.get("f_star")

    # Use best observed f-value as f* approximation when true value is unknown.
    all_f = [v for gn, fv in trajectories.values() for v in fv if np.isfinite(v)]
    f_star_approx = (f_star if f_star is not None
                     else (min(all_f) if all_f else 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    max_k = 0
    ref_anchors: Dict[str, float] = {}   # sname -> f_vals[0] for theory line anchor

    for sname, (grad_norms, f_vals) in trajectories.items():
        if not grad_norms:
            continue
        color  = SOLVER_COLORS.get(sname, "black")
        marker = SOLVER_MARKERS.get(sname, "o")
        ks     = np.arange(len(grad_norms))
        max_k  = max(max_k, len(grad_norms))

        # — gradient norm (log scale) —
        pos_gn = [v for v in grad_norms if v > 0]
        if pos_gn:
            ax1.semilogy(ks, grad_norms, linestyle="-", marker=marker,
                         ms=3, lw=1.8, color=color, label=sname, markevery=5)
        else:
            ax1.plot(ks, grad_norms, linestyle="-", marker=marker,
                     ms=3, lw=1.8, color=color, label=sname, markevery=5)

        # — function excess f(x_k) - f* (log scale) —
        excess = [max(v - f_star_approx, 0.0) for v in f_vals]
        pos_ex = [v for v in excess if v > 0]
        if pos_ex:
            ax2.semilogy(np.arange(len(f_vals)), excess, linestyle="-", marker=marker,
                         ms=3, lw=1.8, color=color, label=sname, markevery=5)
        if f_vals:
            ref_anchors[sname] = max(f_vals[0] - f_star_approx, 1e-30)

    # — theory reference lines on ax2 —
    if max_k > 1:
        ks_ref = np.arange(1, max_k + 1, dtype=float)
        for sname, (exp, ls, rlbl) in THEORY_REFS.items():
            if sname not in ref_anchors:
                continue
            gap0 = ref_anchors[sname]
            ax2.semilogy(ks_ref - 1, gap0 * ks_ref ** (-exp), ls,
                         color=SOLVER_COLORS[sname], lw=1.2, alpha=0.55,
                         label=rlbl)

    # — formatting —
    fstar_lbl = (f"f* = {f_star_approx:.3e}" if f_star is not None
                 else f"f* ≈ {f_star_approx:.3e} (observed)")
    for ax, ylabel, title_sfx in [
        (ax1, r"$\|\nabla f(x_k)\|$",              "Gradient norm"),
        (ax2, f"$f(x_k) - f^*$  ({fstar_lbl})",   "Function excess"),
    ]:
        ax.set_title(f"{label}\n{title_sfx}", fontsize=12)
        ax.set_xlabel("Iteration $k$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = fig_dir / f"convergence_{name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ── performance profile ────────────────────────────────────────────────────────

def plot_performance_profile(csv_rows: List[Dict], fig_dir: Path) -> None:
    """
    Dolan-Moré style performance profile on iteration counts.

    For each problem p and solver s, compute the ratio
        r_{p,s} = iterations_s / min_over_solvers(iterations_s')
    Profile(τ) = fraction of problems where r_{p,s} ≤ τ.
    """
    from collections import defaultdict

    # Group converged runs by problem.
    by_problem: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in csv_rows:
        prob   = row["problem"]
        solver = row["solver"]
        iters  = row["iterations"]
        conv   = str(row["converged"]).lower() == "true"
        if conv and iters != "":
            by_problem[prob][solver] = int(iters)

    if not by_problem:
        print("  Performance profile: no converged runs found, skipping.")
        return

    solvers = sorted({s for d in by_problem.values() for s in d})
    ratios: Dict[str, List[float]] = {s: [] for s in solvers}

    for prob, solver_iters in by_problem.items():
        if not solver_iters:
            continue
        best = min(solver_iters.values())
        for s in solvers:
            if s in solver_iters:
                ratios[s].append(solver_iters[s] / best)
            # Solvers that did not converge are excluded from the profile.

    fig, ax = plt.subplots(figsize=(7, 5))
    tau_max = max((max(v) for v in ratios.values() if v), default=10.0)
    tau_grid = np.linspace(1.0, min(tau_max * 1.1, 50.0), 500)

    for s in solvers:
        if not ratios[s]:
            continue
        r = sorted(ratios[s])
        n_prob = len(r)
        profile = [np.searchsorted(r, t, side="right") / n_prob for t in tau_grid]
        ax.plot(tau_grid, profile,
                color=SOLVER_COLORS.get(s, "black"),
                lw=2, label=s)

    ax.set_xlabel(r"Performance ratio $\tau$", fontsize=12)
    ax.set_ylabel(r"Fraction of problems $\rho_s(\tau)$", fontsize=12)
    ax.set_title("Performance profile (iteration count, converged problems)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(1.0, tau_grid[-1])
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out = fig_dir / "performance_profile.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── CSV writer ─────────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "problem", "solver", "n", "convex", "L3",
        "iterations", "runtime_sec", "final_f", "final_grad_norm",
        "converged", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Building problem suite …")
    suite = build_problem_suite()
    print(f"  {len(suite)} problems defined.\n")

    csv_rows: List[Dict] = []

    for prob in suite:
        pname  = prob["name"]
        n      = len(prob["x0"])
        convex = prob["convex"]
        print(f"── {pname}  (n={n}, convex={convex})")
        trajectories = run_problem(prob, csv_rows)
        for sname, (gn, _) in trajectories.items():
            row = next((r for r in csv_rows
                        if r["problem"] == pname and r["solver"] == sname), {})
            status_str = row.get("status", "?")
            gn_str = f"{gn[-1]:.2e}" if gn else "—"
            print(f"   {sname:<7} iters={len(gn):<4} ‖∇f‖={gn_str}  {status_str}")
        plot_problem(prob, trajectories, FIGURES_DIR)
        print(f"   → figures/convergence_{pname}.pdf\n")

    csv_path = RESULTS_DIR / "convergence_rates.csv"
    write_csv(csv_rows, csv_path)
    print(f"Saved CSV : {csv_path}")

    print("Building performance profile …")
    plot_performance_profile(csv_rows, FIGURES_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
