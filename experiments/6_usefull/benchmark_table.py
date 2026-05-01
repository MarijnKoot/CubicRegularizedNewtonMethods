"""
Benchmark table comparing NCR, ARC, ACRN on all test functions.

Output format mirrors Table 1 from Cartis, Gould & Toint (2011):
  Name | n | NCR (iter, #g, f) | ARC (iter, #g, f) | ACRN (iter, #g, f)

iter = total iterations
#g   = number of accepted (successful) steps
f    = final objective value
"""

import os
import sys
import math

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from methods.NCR import CubicNewton, CRNOptions
from methods.ARC import AdaptiveCubicNewton, ARCParams
from methods.ACRN import AcceleratedCubicNewton
from test_functions.quadratic import quadratic
from test_functions.logsumexp import logsumexp
from test_functions.matrix_scaling import matrix_balancing, matrix_scaling
from test_functions.rosenbrock import rosenbrock

SEED = 42
rng = np.random.default_rng(SEED)

SOLVER_NAMES = ["NCR", "ARC", "ACRN"]

from utilities import estimate_L3, is_convex_at


def run_solvers(f, grad, hess, x0, sigma0=None):
    """Run NCR, ARC, ACRN from x0; return (stats dict, trajectories dict)."""
    if sigma0 is None:
        sigma0 = 1.0
    L3 = estimate_L3(hess, x0)

    results = {}
    trajectories = {}  # solver_name -> list of f values per iteration

    # NCR and ARC share the same .log-based interface.
    # ARC uses step_method="secular" to solve the full cubic subproblem
    # (Hessian-aware Newton-like step) instead of the default Cauchy step
    # (gradient-only, essentially gradient descent → O(κ) iterations).
    for name, Cls, kwargs in [
        ("NCR", CubicNewton,         {"options": CRNOptions(sigma0=sigma0)}),
        ("ARC", AdaptiveCubicNewton, {"params":  ARCParams(sigma0=sigma0),
                                      "step_method": "secular"}),
    ]:
        solver = Cls(f, grad, hess, **kwargs)
        solver.run(x0)
        log = solver.log
        results[name] = {
            "iter": len(log),
            "#g":   sum(1 for e in log if e.get("accepted", True)),
            "f":    log[-1]["f"],
        }
        trajectories[name] = [e["f"] for e in log]

    # ACRN: only valid for convex problems (theory requires PD Hessian).
    # Use the paper's parameter choice (Nesterov 2008, eq. 4.7):
    #   sigma = M = 2*L3   (main-iteration regularization)
    #   N = 12*L3          (estimate-sequence curvature parameter)
    # For quadratics L3≈0 → sigma≈0 → cubic step ≈ Newton step → 1 iteration.
    if is_convex_at(hess, x0):
        acrn = AcceleratedCubicNewton(f, grad, hess, L3=L3)
        x_star, acrn_result = acrn.run(x0)
        history = acrn_result["history"]
        iters = acrn_result["iterations"]
        results["ACRN"] = {
            "iter": iters,
            "#g":   iters,
            "f":    float(f(x_star)),
        }
        trajectories["ACRN"] = [row.f_yk for row in history]
    else:
        results["ACRN"] = None   # non-convex: ACRN not applicable
        trajectories["ACRN"] = None

    return results, trajectories


def fmt_f(val):
    """Format like the paper: mantissa±exp subscript style."""
    if val == 0.0:
        return "0.00"
    exp = math.floor(math.log10(abs(val))) if val != 0 else 0
    if -4 <= exp <= 3:
        return f"{val:.4g}"
    mantissa = val / 10**exp
    sign = "+" if exp >= 0 else ""
    return f"{mantissa:.2f}e{sign}{exp}"


def make_row(name, n, solver_results):
    row = {"Name": name, "n": n}
    for sname in SOLVER_NAMES:
        r = solver_results.get(sname) or {}
        row[f"{sname}_iter"] = r.get("iter", "—")
        row[f"{sname}_#g"]   = r.get("#g",   "—")
        row[f"{sname}_f"]    = fmt_f(r["f"]) if "f" in r else "—"
    return row


def _make_spd(n, cond):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, float(cond), n)
    return Q @ np.diag(eigs) @ Q.T


rows = []
all_trajectories = []  # list of (label, n, trajectories_dict)

# ── Quadratic ──────────────────────────────────────────────────────────────────
for n, cond in [(3, 10), (5, 100), (10, 1e4)]:
    A = _make_spd(n, cond)
    b = rng.normal(size=n)
    f, grad, hess, _ = quadratic(A, b)
    x0 = rng.normal(size=n) * 3.0
    res, traj = run_solvers(f, grad, hess, x0, sigma0=1.0)
    label = f"quadratic (cond={cond:.0e})"
    rows.append(make_row(label, n, res))
    all_trajectories.append((label, n, traj))

# ── Log-Sum-Exp ────────────────────────────────────────────────────────────────
for n, mu in [(5, 1.0), (10, 0.5), (20, 0.2)]:
    m = 4 * n
    a = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    f, grad, hess, M = logsumexp(a, b, mu)
    x0 = np.zeros(n)
    res, traj = run_solvers(f, grad, hess, x0, sigma0=max(float(M), 1e-6))
    label = f"logsumexp (mu={mu})"
    rows.append(make_row(label, n, res))
    all_trajectories.append((label, n, traj))

# ── Matrix Balancing ──────────────────────────────────────────────────────────
for n in [3, 5]:
    A = np.abs(rng.normal(size=(n, n))) + 0.1
    f, grad, hess = matrix_balancing(A)
    x0 = np.zeros(n)
    res, traj = run_solvers(f, grad, hess, x0)
    label = "mat_balancing"
    rows.append(make_row(label, n, res))
    all_trajectories.append((label, n, traj))

# ── Matrix Scaling ────────────────────────────────────────────────────────────
for n in [3, 5]:
    A = np.abs(rng.normal(size=(n, n))) + 0.1
    f, grad, hess = matrix_scaling(A)
    x0 = np.zeros(2 * n)
    res, traj = run_solvers(f, grad, hess, x0)
    label = "mat_scaling"
    rows.append(make_row(label, 2 * n, res))
    all_trajectories.append((label, 2 * n, traj))

# ── Rosenbrock ─────────────────────────────────────────────────────────────────
for N in [2, 4, 10]:
    f, grad, hess = rosenbrock(N)
    x0 = np.zeros(N)
    res, traj = run_solvers(f, grad, hess, x0)
    label = "rosenbrock"
    rows.append(make_row(label, N, res))
    all_trajectories.append((label, N, traj))


# ── Build & display DataFrame ──────────────────────────────────────────────────
col_tuples = [("", "Name"), ("", "n")]
for s in SOLVER_NAMES:
    col_tuples += [(s, "iter"), (s, "#g"), (s, "f")]

df = pd.DataFrame(rows)
df.columns = pd.MultiIndex.from_tuples(col_tuples)

print(df.to_string(index=False))
print()

# Save flat CSV
flat = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(__file__), "benchmark_table.csv")
flat.to_csv(out, index=False)
print(f"Saved: {out}")

# ── Function decrease plots ────────────────────────────────────────────────────
COLORS = {"NCR": "tab:blue", "ARC": "tab:orange", "ACRN": "tab:green"}

n_problems = len(all_trajectories)
ncols = 3
nrows = math.ceil(n_problems / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
axes = np.array(axes).flatten()

for ax, (label, n, traj) in zip(axes, all_trajectories):
    present = [v for vals in traj.values() if vals for v in vals]
    use_log = bool(present) and all(v > 0 for v in present)
    for sname in SOLVER_NAMES:
        vals = traj.get(sname)
        if not vals:
            continue
        if use_log:
            ax.semilogy(range(len(vals)), vals, label=sname, color=COLORS[sname])
        else:
            ax.plot(range(len(vals)), vals, label=sname, color=COLORS[sname])
    ax.set_title(f"{label}\n(n={n})", fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel("f (log scale)" if use_log else "f")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

# Hide unused axes
for ax in axes[n_problems:]:
    ax.set_visible(False)

fig.suptitle("Function decrease over iterations", fontsize=13, y=1.01)
fig.tight_layout()

fig_out = os.path.join(os.path.dirname(__file__), "benchmark_convergence.png")
fig.savefig(fig_out, bbox_inches="tight", dpi=150)
print(f"Saved: {fig_out}")
plt.show()
