"""
Compare the effect of different sigma0 initialisation strategies on NCR and ARC.

Strategies tested
-----------------
fixed_1     : sigma0 = 1.0  (naive constant, current default)
hess_norm   : sigma0 = ||H(x0)||_2  (spectral norm of initial Hessian)
grad_heuristic : sigma0 = ||g(x0)||^(1/3)  (gradient-based proxy for L3 scale)
L3_est      : sigma0 = L3  (finite-difference Hessian-Lipschitz estimate)

For each (problem, sigma_strategy, solver) triple we record:
  - total iterations
  - number of accepted steps (#g)
  - final f value
  - full f-vs-iteration trajectory

Output: table printed to stdout + convergence plot saved to
        experiments/sigma_comparison.png
"""

import os
import sys
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC  = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from methods.NCR import CubicNewton, CRNOptions
from methods.ARC import AdaptiveCubicNewton, ARCParams
from test_functions.quadratic      import quadratic
from test_functions.logsumexp      import logsumexp
from test_functions.matrix_scaling import matrix_balancing, matrix_scaling
from test_functions.rosenbrock     import rosenbrock

SEED = 42
rng  = np.random.default_rng(SEED)

MAX_ITER = 200

# ── Sigma strategies ──────────────────────────────────────────────────────────

def _estimate_L3(hess, x0, eps=1e-4, n_dirs=5):
    H0      = hess(x0)
    est_rng = np.random.default_rng(0)
    L3 = 0.0
    for _ in range(n_dirs):
        v  = est_rng.normal(size=len(x0))
        v /= np.linalg.norm(v)
        H1 = hess(x0 + eps * v)
        L3 = max(L3, np.linalg.norm(H1 - H0, "fro") / eps)
    return max(L3, 1e-6)


STRATEGIES = {
    "fixed_1":        lambda grad, hess, x0: 1.0,
    "hess_norm":      lambda grad, hess, x0: float(
                          np.linalg.norm(
                              0.5*(np.asarray(hess(x0))+np.asarray(hess(x0)).T),
                              ord=2
                          )
                      ),
    "grad_heuristic": lambda grad, hess, x0: max(
                          float(np.linalg.norm(grad(x0)))**(1/3), 1e-6
                      ),

                      
    "L3_est":         lambda grad, hess, x0: _estimate_L3(hess, x0),
}

STRATEGY_LABELS = {
    "fixed_1":        r"$\sigma_0=1$",
    "hess_norm":      r"$\sigma_0=\|H_0\|_2$",
    "grad_heuristic": r"$\sigma_0=\|\nabla f_0\|^{1/3}$",
    "L3_est":         r"$\sigma_0=\hat{L}_3$",
}

SOLVERS      = ["NCR", "ARC"]
SOLVER_COLORS = {"NCR": "tab:blue", "ARC": "tab:orange"}
STRATEGY_LS   = {
    "fixed_1":        "-",
    "hess_norm":      "--",
    "grad_heuristic": "-.",
    "L3_est":         ":",
}

# ── Problem suite ─────────────────────────────────────────────────────────────

def _make_spd(n, cond):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eigs = np.linspace(1.0, float(cond), n)
    return Q @ np.diag(eigs) @ Q.T


problems = []   # list of (label, n, f, grad, hess, x0)

for n, cond in [(3, 10), (5, 100), (10, 1e4)]:
    A = _make_spd(n, cond)
    b = rng.normal(size=n)
    f, grad, hess, _ = quadratic(A, b)
    x0 = rng.normal(size=n) * 3.0
    problems.append((f"quadratic (cond={cond:.0e})", n, f, grad, hess, x0))

for n, mu in [(5, 1.0), (10, 0.5), (20, 0.2)]:
    m = 4 * n
    a = rng.normal(size=(m, n))
    b = rng.normal(size=m)
    f, grad, hess, _ = logsumexp(a, b, mu)
    x0 = np.zeros(n)
    problems.append((f"logsumexp (mu={mu})", n, f, grad, hess, x0))

for n in [3, 5]:
    A = np.abs(rng.normal(size=(n, n))) + 0.1
    f, grad, hess = matrix_balancing(A)
    x0 = np.zeros(n)
    problems.append(("mat_balancing", n, f, grad, hess, x0))

for n in [3, 5]:
    A = np.abs(rng.normal(size=(n, n))) + 0.1
    f, grad, hess = matrix_scaling(A)
    x0 = np.zeros(2 * n)
    problems.append(("mat_scaling", 2 * n, f, grad, hess, x0))

for N in [2, 4, 10]:
    f, grad, hess = rosenbrock(N)
    x0 = np.zeros(N)
    problems.append(("rosenbrock", N, f, grad, hess, x0))


# ── Run one solver with one sigma strategy ───────────────────────────────────

def run_one(solver_name, sigma0, f, grad, hess, x0):
    if solver_name == "NCR":
        s = CubicNewton(f, grad, hess,
                        options=CRNOptions(sigma0=sigma0, max_iter=MAX_ITER))
    else:
        s = AdaptiveCubicNewton(f, grad, hess,
                                params=ARCParams(sigma0=sigma0, max_iter=MAX_ITER),
                                step_method="secular")
    s.run(x0)
    log = s.log
    return {
        "iter":  len(log),
        "#g":    sum(1 for e in log if e.get("accepted", True)),
        "f":     log[-1]["f"],
        "traj":  [e["f"] for e in log],
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

rows = []
all_results = []   # (label, n, solver, strategy, sigma0_val, result)

for (label, n, f, grad, hess, x0) in problems:
    prob_row = {"Name": label, "n": n}
    prob_results = {}
    for strat_name, strat_fn in STRATEGIES.items():
        sigma0_val = strat_fn(grad, hess, x0)
        prob_results[strat_name] = {}
        for solver_name in SOLVERS:
            res = run_one(solver_name, sigma0_val, f, grad, hess, x0)
            prob_results[strat_name][solver_name] = res
            all_results.append((label, n, solver_name, strat_name, sigma0_val, res))
            prob_row[f"{solver_name}_{strat_name}_iter"] = res["iter"]
            prob_row[f"{solver_name}_{strat_name}_f"]    = res["f"]
    rows.append(prob_row)

# ── Print compact summary table ───────────────────────────────────────────────

print(f"\n{'Problem':<28} {'n':>3}  "
      + "  ".join(f"{s}({st[:5]})" for s in SOLVERS for st in STRATEGIES))
print("-" * 110)
for row in rows:
    label = row["Name"]
    n     = row["n"]
    cells = []
    for s in SOLVERS:
        for st in STRATEGIES:
            it = row[f"{s}_{st}_iter"]
            cells.append(f"{it:>4}")
    print(f"{label:<28} {n:>3}  " + "  ".join(cells))

print("\nColumns: iter count per (solver × strategy)")
print("Strategies:", list(STRATEGIES.keys()))

# ── Save detailed CSV ─────────────────────────────────────────────────────────

detail_rows = []
for label, n, solver, strat, sigma0_val, res in all_results:
    detail_rows.append({
        "problem": label, "n": n,
        "solver": solver, "strategy": strat,
        "sigma0": sigma0_val,
        "iter": res["iter"], "#g": res["#g"], "f": res["f"],
    })
csv_path = os.path.join(ROOT, "experiments", "sigma_comparison.csv")
pd.DataFrame(detail_rows).to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ── Convergence plots ─────────────────────────────────────────────────────────
# One row per problem, two columns (NCR / ARC).
# Each subplot shows f vs iteration for all 4 strategies.

n_prob = len(problems)
fig, axes = plt.subplots(n_prob, 2, figsize=(12, 3.5 * n_prob), squeeze=False)

for row_idx, (label, n, f, grad, hess, x0) in enumerate(problems):
    for col_idx, solver_name in enumerate(SOLVERS):
        ax = axes[row_idx][col_idx]
        has_data = False
        for strat_name in STRATEGIES:
            # find corresponding result
            traj = next(
                r["traj"]
                for lbl, nn, sv, st, _, r in all_results
                if lbl == label and nn == n and sv == solver_name and st == strat_name
            )
            if not traj:
                continue
            vals = traj
            # use log scale if all positive
            use_log = all(v > 0 for v in vals)
            xs = range(len(vals))
            ls = STRATEGY_LS[strat_name]
            lbl_str = STRATEGY_LABELS[strat_name]
            if use_log:
                ax.semilogy(xs, vals, ls, color=SOLVER_COLORS[solver_name],
                            label=lbl_str, alpha=0.85)
            else:
                ax.plot(xs, vals, ls, color=SOLVER_COLORS[solver_name],
                        label=lbl_str, alpha=0.85)
            has_data = True

        ax.set_title(f"{label} (n={n})  —  {solver_name}", fontsize=8)
        ax.set_xlabel("iteration", fontsize=8)
        ax.set_ylabel("f", fontsize=8)
        ax.tick_params(labelsize=7)
        if has_data:
            ax.legend(fontsize=7)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)

fig.suptitle("Convergence under different $\\sigma_0$ strategies\n"
             "(linestyle = strategy, colour = solver NCR/ARC)",
             fontsize=11, y=1.002)
fig.tight_layout()

fig_path = os.path.join(ROOT, "experiments", "sigma_comparison.png")
fig.savefig(fig_path, bbox_inches="tight", dpi=150)
print(f"Saved: {fig_path}")
