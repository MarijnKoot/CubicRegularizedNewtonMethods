"""
Sigma sensitivity study for cubic-regularization solvers.

For a chosen test function and fixed starting point, runs NCR, ARC, and ACRN
with three sigma values:
  - sigma_small  = sigma_ref × SIGMA_MULTIPLIERS[0]   (under-regularized)
  - sigma_ref    = 2 × estimate_L3(hess, x0)          (natural / paper prescription)
  - sigma_large  = sigma_ref × SIGMA_MULTIPLIERS[2]   (over-regularized)

Produces two figures per solver:
  1. Optimization paths on a contour plot
  2. Convergence curves (f(x_k) vs iteration)
"""







import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from utilities import estimate_L3, is_convex_at
from test_functions.logsumexp import logsumexp
from test_functions.quadratic import quadratic
from test_functions.matrix_scaling import matrix_balancing
from test_functions.rosenbrock import rosenbrock
from test_functions.cubic_poly import cubic_poly
from test_functions.quartic_convex import quartic_convex
from test_functions.cubic_norm import cubic_norm
from methods.NCR import CubicNewton, CRNOptions
from methods.ACRN import AcceleratedCubicNewton
from methods.ARC import AdaptiveCubicNewton, ARCParams


# ── Settings ───────────────────────────────────────────────────────────────────

FUNCTION_NAME     = "cubic_norm"   # key in FUNCTION_CONFIGS below
SIGMA_STUDY_X0    = np.array([2.0, -1.5])
SIGMA_MULTIPLIERS = [0.0001, 1.0, 10000.0]  # small / reference / large
SOLVERS           = ["NCR", "ARC", "ACRN"]

MAX_ITER = 300
TOL_GRAD = 1e-6

# When True, the reference sigma (×1 multiplier) is pinned for NCR/ACRN if the
# exact L3 is known — sigma will not change during the run.
# When False, solvers always adapt sigma even when starting from the theoretical value.
FIXED_SIGMA_ON_THEORY = True

# Theoretical convergence rates shown as dashed reference lines in convergence plots.
# Each entry: exponent p  →  reference line  f* + C·k^{-p}
THEORY_REFS = {
    "NCR":  (2, "tab:gray", r"$k^{-2}$"),
    "ARC":  (2, "tab:gray", r"$k^{-2}$"),
    "ACRN": (3, "black",    r"$k^{-3}$"),
}

# Human-readable color names for the summary table (matched to matplotlib default cycle)
_SIGMA_COLOR_NAMES = ["blue", "orange", "green", "red", "purple"]


# ── Test function configs ───────────────────────────────────────────────────────

def _make_function_configs():
    a_lse = np.array([[1, 0], [0, 1], [-1, -1]], dtype=float)
    f_lse, g_lse, h_lse, M_lse = logsumexp(a_lse, np.zeros(3), mu=0.5)

    A_qd = np.array([[4.0, 1.0], [1.0, 3.0]])
    f_qd, g_qd, h_qd, L3_qd = quadratic(A_qd, np.array([1.0, 2.0]))

    f_rb, g_rb, h_rb = rosenbrock(N=2)

    A_mb = np.array([[1.0, 0.5], [0.25, 1.0]])
    f_mb, g_mb, h_mb = matrix_balancing(A_mb)

    Q_cp = np.array([[2.0, 0.5], [0.5, 1.0]])
    f_cp, g_cp, h_cp, L3_cp = cubic_poly(Q_cp, gamma=1.0)

    f_qv, g_qv, h_qv = quartic_convex(n=2)

    f_cn, g_cn, h_cn, L3_cn = cubic_norm(n=2)

    return {
        "logsumexp":        dict(f=f_lse, grad=g_lse, hess=h_lse,
                                 xlim=(-3, 3), ylim=(-3, 3)),
        "quadratic":        dict(f=f_qd, grad=g_qd, hess=h_qd, L3=L3_qd,
                                 xlim=(-4, 4), ylim=(-4, 4)),
        "rosenbrock":       dict(f=f_rb, grad=g_rb, hess=h_rb,
                                 xlim=(-1, 3), ylim=(-2, 5)),
        "matrix_balancing": dict(f=f_mb, grad=g_mb, hess=h_mb,
                                 xlim=(-3, 3), ylim=(-3, 3)),
        "cubic_poly":       dict(f=f_cp, grad=g_cp, hess=h_cp, L3=L3_cp,
                                 xlim=(-3, 2.5), ylim=(-2.5, 3)),
        "quartic_convex":   dict(f=f_qv, grad=g_qv, hess=h_qv,
                                 xlim=(-3, 2.5), ylim=(-2.5, 3)),
        "cubic_norm":       dict(f=f_cn, grad=g_cn, hess=h_cn, L3=L3_cn,
                                 xlim=(-3, 3), ylim=(-3, 3)),
    }


# ── Solver construction ────────────────────────────────────────────────────────

def _make_solver(name, f, grad, hess, sigma_val, x0, L3_known=None):
    """Build a solver with the requested starting sigma for sensitivity analysis."""
    s = max(float(sigma_val), 1e-15)
    # Pin sigma only when the exact L3 is known AND the global toggle is enabled.
    fixed = (L3_known is not None) and FIXED_SIGMA_ON_THEORY

    if name == "NCR":
        return CubicNewton(f, grad, hess, options=CRNOptions(
            sigma0=s,
            sigma_min=s   if fixed else 1e-15,
            sigma_max=s   if fixed else 5e11,
            tol_grad=TOL_GRAD, tol_step=1e-9, max_iter=MAX_ITER))
    if name == "ARC":
        return AdaptiveCubicNewton(f=f, grad=grad, hess=hess,
            params=ARCParams(sigma0=s,
                             sigma_min=1e-15,
                             sigma_max=5e11,
                             eta1=0.1, eta2=0.9,
                             tol_grad=TOL_GRAD, max_iter=MAX_ITER),
            step_method="secular")
    if name == "ACRN":
        L3 = L3_known if L3_known is not None else estimate_L3(hess, x0)
        sigma_ref = max(2.0 * float(L3), 1e-15)
        adaptive = (not fixed) or (not np.isclose(s, sigma_ref, rtol=1e-12, atol=1e-15))
        return AcceleratedCubicNewton(f=f, grad=grad, hess=hess,
            L3=max(L3, 1e-15), sigma=s,
            tol_grad=TOL_GRAD, tol_step=1e-12, max_iter=MAX_ITER,
            verbose=False, adaptive_sigma=adaptive,
            sigma_min=1e-15 if adaptive else s,
            sigma_max=5e11 if adaptive else s)
    raise ValueError(f"Unknown solver: {name}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_path(solver, run_result):
    if isinstance(run_result, tuple):
        x_final, info = run_result
        rows = info.get("history", [])
        path = np.array([r.xk for r in rows], dtype=float) if rows else None
    else:
        x_final = run_result
        xs = [e["x"] for e in getattr(solver, "log", []) if "x" in e]
        path = np.array(xs, dtype=float) if xs else None
    return x_final, path


def _format_ax(ax, title, xlabel, ylabel, fontsize=22, legend_loc="best"):
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=16)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=16, loc=legend_loc)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    configs = _make_function_configs()
    cfg = configs[FUNCTION_NAME]
    f, grad, hess = cfg["f"], cfg["grad"], cfg["hess"]
    xlim, ylim = cfg["xlim"], cfg["ylim"]
    x0 = np.asarray(SIGMA_STUDY_X0, dtype=float)

    # Reference sigma: 2 × L3 (exact if available, otherwise estimated numerically)
    L3_est    = cfg["L3"] if cfg.get("L3") is not None else estimate_L3(hess, x0)
    sigma_ref = max(2.0 * L3_est, 1e-6)
    sigma_vals = [sigma_ref * m for m in SIGMA_MULTIPLIERS]

    labels = [f"σ = {v:.2g} (×{m})" for v, m in zip(sigma_vals, SIGMA_MULTIPLIERS)]
    print(f"\nFunction : {FUNCTION_NAME}")
    print(f"L3 estimate : {L3_est:.4g}")
    print(f"sigma_ref   : {sigma_ref:.4g}")
    for lbl, sv in zip(labels, sigma_vals):
        print(f"  {lbl}  →  {sv:.4g}")

    # Contour grid
    xs = np.linspace(*xlim, 200)
    ys = np.linspace(*ylim, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda xi, yi: f(np.array([xi, yi])))(X, Y)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_solvers = len(SOLVERS)

    # Reserve a dedicated narrow column for the colorbar so it never overlaps the plots
    from matplotlib.gridspec import GridSpec
    fig_p = plt.figure(figsize=(6 * n_solvers + 1, 6))
    gs    = GridSpec(1, n_solvers + 1,
                     width_ratios=[1] * n_solvers + [0.05],
                     wspace=0.3, figure=fig_p)
    axes_p = np.array([fig_p.add_subplot(gs[0, i]) for i in range(n_solvers)])
    for ax in axes_p[1:]:          # share axes manually
        ax.sharex(axes_p[0])
        ax.sharey(axes_p[0])
    cbar_ax = fig_p.add_subplot(gs[0, -1])

    fig_c, axes_c = plt.subplots(1, n_solvers, figsize=(6 * n_solvers, 5))
    axes_c = np.atleast_1d(axes_c)

    fig_s, axes_s = plt.subplots(1, n_solvers, figsize=(6 * n_solvers, 5))
    axes_s = np.atleast_1d(axes_s)

    # Collect stats for the summary table: (solver, sigma_label, color) -> stats dict
    table_rows = []

    for col_idx, sname in enumerate(SOLVERS):
        ax_p = axes_p[col_idx]
        ax_c = axes_c[col_idx]
        ax_s = axes_s[col_idx]
        cf = ax_p.contourf(X, Y, Z, levels=50)
        # Matplotlib API compatibility: QuadContourSet.collections is absent in newer versions.
        if hasattr(cf, "set_rasterized"):
            cf.set_rasterized(True)
        elif hasattr(cf, "collections"):
            for coll in cf.collections:
                coll.set_rasterized(True)
        ax_p.contour(X, Y, Z, levels=50, colors="white", linewidths=0.4, alpha=0.3)

        if sname == "ACRN" and not is_convex_at(hess, x0):
            ax_p.text(0.5, 0.5, "ACRN N/A\n(nonconvex at x0)",
                      transform=ax_p.transAxes, ha="center", va="center", fontsize=14)
            _format_ax(ax_p, sname, "$x_1$", "$x_2$")
            _format_ax(ax_c, sname, "iteration", "$f(x_k)$", legend_loc="upper right")
            _format_ax(ax_s, sname, "iteration", r"$\sigma_k$", legend_loc="upper right")
            continue

        ref_f_vals  = None   # f_vals for the middle (reference) sigma — used to anchor theory lines
        all_f_min   = np.inf
        max_iters_c = 0

        for sig_idx, (sigma_val, lbl) in enumerate(zip(sigma_vals, labels)):
            color = colors[sig_idx % len(colors)]
            # NCR/ACRN: pin sigma at the reference multiplier (×1) to the theoretical value.
            # ARC: always adapt — its ratio-based σ-update is independent of L3 by design.
            is_ref = (sig_idx == 1)
            l3_for_solver = cfg.get("L3") if (is_ref and sname != "ARC") else None
            solver = _make_solver(sname, f, grad, hess, sigma_val, x0, l3_for_solver)
            t0 = time.perf_counter()
            raw = solver.run(x0.copy())
            elapsed = getattr(solver, "runtime", None) or (time.perf_counter() - t0)
            x_final, path = _extract_path(solver, raw)

            if path is None:
                path = x0[np.newaxis, :]
            if not np.allclose(path[0], x0):
                path = np.vstack([x0, path])
            if not np.allclose(path[-1], x_final):
                path = np.vstack([path, x_final])

            if hasattr(solver, "log") and solver.log:
                f_vals = [e["f"] for e in solver.log] + [f(x_final)]
            elif isinstance(raw, tuple) and "history" in raw[1]:
                f_vals = [float(f(r.xk)) for r in raw[1]["history"]] + [f(x_final)]
            else:
                f_vals = [f(x_final)]

            iters    = len(f_vals) - 1
            rejected = sum(1 for e in getattr(solver, "log", []) if not e.get("accepted", True))
            f_final  = float(f(x_final))
            x_str    = "[" + ", ".join(f"{v:.4f}" for v in x_final) + "]"
            grad_norm = float(np.linalg.norm(grad(x_final)))

            # track data for theory reference lines
            finite_vals = [v for v in f_vals if np.isfinite(v)]
            if finite_vals:
                all_f_min = min(all_f_min, min(finite_vals))
            max_iters_c = max(max_iters_c, len(f_vals))
            if sig_idx == 1:
                ref_f_vals = f_vals

            table_rows.append({
                "Solver":       sname,
                "σ":            f"{sigma_val:.2g}",
                "Iter":         iters,
                "Rejected":     rejected,
                "Time (s)":     f"{elapsed:.4f}",
                "grad_norm":    f"{grad_norm:.2e}",
                "f(x*)":        f"{f_final:.4e}",
                "x*":           x_str,
                "_color_name":  _SIGMA_COLOR_NAMES[sig_idx % len(_SIGMA_COLOR_NAMES)],
            })
            print(f"  {sname}  {lbl}  iters={iters}  rejected={rejected}"
                  f"  time={elapsed:.4f}s  ||grad||={grad_norm:.2e}  f={f_final:.4e}  x*={x_str}")

            ax_p.plot(path[:, 0], path[:, 1], "-o", ms=5, lw=2, color=color,
                      label=f"$\\sigma$={sigma_val:.2g} ({iters} iter)")
            ax_p.scatter(path[0, 0], path[0, 1], s=100, marker="o",
                         color=color, edgecolors="black", zorder=5)
            ax_p.scatter(path[-1, 0], path[-1, 1], s=120, marker="x",
                         color=color, linewidths=3, zorder=5)

            lbl_c = f"$\\sigma_0$={sigma_val:.2g}"
            if all(v > 0 for v in f_vals):
                ax_c.semilogy(range(len(f_vals)), f_vals, "-o", ms=4, lw=2,
                              color=color, label=lbl_c)
            else:
                ax_c.plot(range(len(f_vals)), f_vals, "-o", ms=4, lw=2,
                          color=color, label=lbl_c)

            # σ_k evolution: read from log if available, else flat (ACRN fixed sigma)
            if hasattr(solver, "log") and solver.log and "sigma" in solver.log[0]:
                sigma_hist = [e["sigma"] for e in solver.log]
            else:
                sigma_hist = [sigma_val] * iters
            ax_s.semilogy(range(len(sigma_hist)), sigma_hist, "-o", ms=4, lw=2,
                          color=color, label=lbl_c)

        # Theory reference line anchored at the first f-value of the reference sigma
        if sname in THEORY_REFS and ref_f_vals and np.isfinite(all_f_min):
            exp, rcol, rlbl = THEORY_REFS[sname]
            gap0 = max(ref_f_vals[0] - all_f_min, 1e-30)
            ks = np.arange(1, max_iters_c + 1, dtype=float)
            ax_c.semilogy(ks - 1, all_f_min + gap0 * ks ** (-exp), "--",
                          color=rcol, lw=1.5, alpha=0.7, label=rlbl)

        _format_ax(ax_p, sname, "$x_1$", "$x_2$")
        _format_ax(ax_c, sname, "iteration", "$f(x_k)$", legend_loc="upper right")
        _format_ax(ax_s, sname, "iteration", r"$\sigma_k$", legend_loc="upper right")

    # Enforce configured limits — paths that wander outside would otherwise
    # auto-expand the shared axes and push the contourf out of view
    axes_p[0].set_xlim(xlim)
    axes_p[0].set_ylim(ylim)

    # ── Print booktabs LaTeX table to terminal ─────────────────────────────────
    # Requires \usepackage[table]{xcolor} and \usepackage{booktabs} in LaTeX preamble
    col_keys = ["Solver", r"$\sigma$", "Iter", "Rej.", "Time (s)",
                r"$\|\nabla f\|$", r"$f(x^*)$", r"$x^*$", "Color"]
    col_fmt  = "ll" + "r" * (len(col_keys) - 3) + "l"

    lines = []
    prev_solver = None
    for r in table_rows:
        if r["Solver"] != prev_solver and prev_solver is not None:
            lines.append(r"    \midrule")
        solver_cell = r["Solver"] if r["Solver"] != prev_solver else ""
        lines.append(
            f"    {solver_cell} & {r['σ']} & {r['Iter']} & {r['Rejected']} & "
            f"{r['Time (s)']} & {r['grad_norm']} & {r['f(x*)']} & "
            f"{r['x*']} & {r['_color_name']} \\\\"
        )
        prev_solver = r["Solver"]

    header = " & ".join(col_keys) + r" \\"
    tex = (
        r"\begin{table}[ht]" + "\n"
        r"  \centering" + "\n"
        f"  \\caption{{$\\sigma$ sensitivity — {FUNCTION_NAME}}}\n"
        f"  \\label{{tab:sigma-{FUNCTION_NAME}}}\n"
        f"  \\begin{{tabular}}{{{col_fmt}}}\n"
        r"    \toprule" + "\n"
        f"    {header}\n"
        r"    \midrule" + "\n"
        + "\n".join(lines) + "\n"
        + r"    \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}"
    )
    print("\n% ── LaTeX sigma-sensitivity table " + "─" * 40)
    print("% Requires: \\usepackage[table]{xcolor}, \\usepackage{booktabs}")
    print(tex)

    # Colorbar in its dedicated axes — rasterize fills so colors survive PGF/PDF export
    cbar = fig_p.colorbar(cf, cax=cbar_ax)
    cbar.set_label("$f(x)$", fontsize=22)
    cbar.ax.tick_params(labelsize=22)
    if hasattr(cbar, "solids") and hasattr(cbar.solids, "set_rasterized"):
        cbar.solids.set_rasterized(True)

    # Use ASCII-safe titles (em dash -> "--" avoids LaTeX encoding issues in PGF)
    fig_p.suptitle(f"$\\sigma$ sensitivity -- paths ({FUNCTION_NAME})", fontsize=26)
    fig_p.subplots_adjust(top=0.84, bottom=0.10, left=0.06, right=0.88, wspace=0.3)

    fig_c.suptitle(f"$\\sigma$ sensitivity -- convergence ({FUNCTION_NAME})", fontsize=26)
    fig_c.tight_layout()
    fig_s.tight_layout()

    # ── Export ─────────────────────────────────────────────────────────────────
    fig_dir = os.path.join(ROOT, "figures", "sigma_study")
    os.makedirs(fig_dir, exist_ok=True)
    fig_p.savefig(os.path.join(fig_dir, f"{FUNCTION_NAME}_paths2.pdf"))
    print(f"Saved PDF: {fig_dir}/{FUNCTION_NAME}_paths.pdf")
    fig_c.savefig(os.path.join(fig_dir, f"{FUNCTION_NAME}_convergence2.pdf"))
    print(f"Saved PDF: {fig_dir}/{FUNCTION_NAME}_convergence.pdf")
    fig_s.savefig(os.path.join(fig_dir, f"{FUNCTION_NAME}_sigma_evolution2.pdf"))
    print(f"Saved PDF: {fig_dir}/{FUNCTION_NAME}_sigma_evolution.pdf")

    plt.show()


if __name__ == "__main__":
    main()
