import os
import sys

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
from methods.NCR import CubicNewton, CRNOptions
from methods.ACRN import AcceleratedCubicNewton
from methods.ARC import AdaptiveCubicNewton, ARCParams
from methods.pure_newton import PureNewton, PureNewtonOptions


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_path(solver, run_result):
    """Return (x_final, path_array | None) from a solver and its run() output."""
    if isinstance(run_result, tuple):
        x_final, info = run_result
        rows = info.get("history", [])
        path = np.array([r.xk for r in rows], dtype=float) if rows else None
    else:
        x_final = run_result
        xs = [e["x"] for e in getattr(solver, "log", []) if "x" in e]
        path = np.array(xs, dtype=float) if xs else None
    return x_final, path


def _format_ax(ax, title, xlabel, ylabel, fontsize=18):
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


def _show_or_close(fig, title, show):
    if show:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        plt.show()
    else:
        plt.close(fig)


# ── Solver factory ─────────────────────────────────────────────────────────────

def _make_solvers(f, grad, hess, sigma, arc_m0_scale=1.0, L3_known=None):
    sigma0 = max(float(sigma), 1e-15)

    def _acrn_factory(x0):
        L3 = L3_known if L3_known is not None else estimate_L3(hess, x0)
        return AcceleratedCubicNewton(
            f=f, grad=grad, hess=hess,
            L3=max(L3, 1e-15),
            sigma=max(2.0 * L3, 1e-15),
            tol_grad=1e-6, tol_step=1e-12, max_iter=300, verbose=False,
            adaptive_sigma=(L3_known is None),
        )

    return [
        ("Newton", lambda: PureNewton(f, grad, hess, options=PureNewtonOptions(tol_grad=1e-6, max_iter=150))),
        ("NCR",    lambda: CubicNewton(f, grad, hess, options=CRNOptions(
                       sigma0=sigma0, sigma_min=sigma0,
                       tol_grad=1e-6, tol_step=1e-9, max_iter=150))),
        ("ACRN",   _acrn_factory),
        ("ARC",    lambda: AdaptiveCubicNewton(f=f, grad=grad, hess=hess,
                       params=ARCParams(sigma0=sigma0 * arc_m0_scale, sigma_min=sigma0,
                                        eta1=0.1, eta2=0.9, gamma1=2.0, gamma2=4.0,
                                        tol_grad=1e-6, max_iter=150),
                       step_method="secular")),
    ]


# ── Test function configs ──────────────────────────────────────────────────────

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

    return {
        "logsumexp":        dict(f=f_lse, grad=g_lse, hess=h_lse, sigma=float(M_lse),
                                 xlim=(-3, 3),   ylim=(-3, 3),   arc_m0_scale=10.0),
        "quadratic":        dict(f=f_qd, grad=g_qd, hess=h_qd, sigma=0.0, L3=L3_qd,
                                 xlim=(-4, 4),   ylim=(-4, 4),   arc_m0_scale=1.0),
        "rosenbrock":       dict(f=f_rb, grad=g_rb, hess=h_rb, sigma=1.0,
                                 xlim=(-3, 3),   ylim=(-2, 7),   arc_m0_scale=1.0),
        "matrix_balancing": dict(f=f_mb, grad=g_mb, hess=h_mb, sigma=1.0,
                                 xlim=(-3, 3),   ylim=(-3, 3),   arc_m0_scale=1.0),
        "cubic_poly":       dict(f=f_cp, grad=g_cp, hess=h_cp, sigma=L3_cp, L3=L3_cp,
                                 xlim=(-3, 2.5), ylim=(-2.5, 3), arc_m0_scale=1.0),
        "quartic_convex":   dict(f=f_qv, grad=g_qv, hess=h_qv, sigma=2.0,
                                 xlim=(-3, 2.5), ylim=(-2.5, 3), arc_m0_scale=1.0),
    }


# ── Settings ───────────────────────────────────────────────────────────────────

FUNCTION_NAME    = "rosenbrock"
SHOW_PATHS       = True
SHOW_CONVERGENCE = False # True
SHOW_RATE        = False # True

PARAM_SETS = [
    {"name": "x0_a", "x0": np.array([2.0, -1.5])},
    {"name": "x0_b", "x0": np.array([-2.5, 2.0])},
    {"name": "x0_c", "x0": np.array([1.0, 2.5])},
]

RATE_REFS = [
    (2, "tab:gray", "$k^{-2}$ (NCR)"),
    (3, "black",    "$k^{-3}$ (ACRN theory)"),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    configs = _make_function_configs()
    cfg = configs[FUNCTION_NAME]
    f, grad, hess = cfg["f"], cfg["grad"], cfg["hess"]
    xlim, ylim = cfg["xlim"], cfg["ylim"]

    solver_defs = _make_solvers(f, grad, hess, cfg["sigma"],
                                arc_m0_scale=cfg["arc_m0_scale"],
                                L3_known=cfg.get("L3"))
    cols   = len(solver_defs)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Contour grid (shared across all plots)
    x = np.linspace(*xlim, 200)
    y = np.linspace(*ylim, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda xi, yi: f(np.array([xi, yi])))(X, Y)

    # ── Run all solvers × starting points ─────────────────────────────────────
    conv_data  = {name: {} for name, _ in solver_defs}  # name -> {pname -> f_vals}
    path_data  = {name: {} for name, _ in solver_defs}  # name -> {pname -> path array}
    final_data = {name: {} for name, _ in solver_defs}  # name -> {pname -> x_final}

    for name, solver_factory in solver_defs:
        for param in PARAM_SETS:
            pname = param["name"]
            x0    = np.asarray(param["x0"], dtype=float)

            if name == "ACRN" and not is_convex_at(hess, x0):
                conv_data[name][pname] = path_data[name][pname] = final_data[name][pname] = None
                print(f"{name} {pname}: skipped (nonconvex at x0)")
                continue

            solver  = solver_factory(x0) if name == "ACRN" else solver_factory()
            raw     = solver.run(x0.copy())
            x_final, path = _extract_path(solver, raw)

            reason = getattr(solver, "termination_reason", None)
            if reason is None and isinstance(raw, tuple):
                reason = raw[1].get("note", "max_iter")
            print(f"{name} {pname}: {reason}")

            # Ensure path starts at x0 and ends at x_final
            if path is None:
                path = x0[np.newaxis, :]
            if not np.allclose(path[0], x0):
                path = np.vstack([x0, path])
            if not np.allclose(path[-1], x_final):
                path = np.vstack([path, x_final])

            # f-value trace for convergence plots
            if hasattr(solver, "log") and solver.log:
                f_vals = [e["f"] for e in solver.log] + [f(x_final)]
            elif isinstance(raw, tuple) and "history" in raw[1]:
                f_vals = [float(f(r.xk)) for r in raw[1]["history"]] + [f(x_final)]
            else:
                f_vals = [f(x_final)]

            path_data[name][pname]  = path
            final_data[name][pname] = x_final
            conv_data[name][pname]  = f_vals

    # ── Paths plot ─────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, cols, figsize=(6 * cols, 5), sharex=True, sharey=True)
    axes1 = np.atleast_1d(axes1)
    for col_idx, (name, _) in enumerate(solver_defs):
        ax = axes1[col_idx]
        ax.contourf(X, Y, Z, levels=50)
        for pt_idx, param in enumerate(PARAM_SETS):
            pname  = param["name"]
            path   = path_data[name].get(pname)
            color  = colors[pt_idx % len(colors)]
            x0     = np.asarray(param["x0"], dtype=float)
            if path is None:
                ax.scatter(x0[0], x0[1], s=120, marker="o", color=color,
                           edgecolors="black", zorder=5)
                ax.text(x0[0], x0[1], f" {pname} N/A", fontsize=10, color=color, zorder=6)
                continue
            f_vals = conv_data[name].get(pname, [])
            iters  = len(f_vals) - 1 if f_vals else len(path) - 1
            ax.plot(path[:, 0], path[:, 1], "-o", ms=6, lw=2.5, color=color,
                    label=f"{pname} (iters={iters})")
            ax.scatter(path[0, 0], path[0, 1], s=120, marker="o", color=color,
                       edgecolors="black", zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], s=150, marker="x", color=color,
                       linewidths=3, zorder=5)
        _format_ax(ax, name, "$x_1$", "$x_2$")
    _show_or_close(fig1, f"Optimization paths — {FUNCTION_NAME}", SHOW_PATHS)

    # ── Convergence plot ───────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, cols, figsize=(6 * cols, 5), sharey=False)
    axes2 = np.atleast_1d(axes2)
    for col_idx, (name, _) in enumerate(solver_defs):
        ax = axes2[col_idx]
        for pt_idx, param in enumerate(PARAM_SETS):
            pname = param["name"]
            vals  = conv_data[name].get(pname)
            if not vals:
                continue
            color = colors[pt_idx % len(colors)]
            if all(v > 0 for v in vals):
                ax.semilogy(range(len(vals)), vals, "-o", ms=4, lw=2, color=color, label=pname)
            else:
                ax.plot(range(len(vals)), vals, "-o", ms=4, lw=2, color=color, label=pname)
        _format_ax(ax, name, "iteration", "$f(x_k)$")
    _show_or_close(fig2, f"Convergence — {FUNCTION_NAME}", SHOW_CONVERGENCE)

    # ── Rate verification plot (log-log) ───────────────────────────────────────
    all_vals = [v for traces in conv_data.values()
                for vals in traces.values() if vals for v in vals]
    f_star = min(all_vals) if all_vals else 0.0

    fig3, axes3 = plt.subplots(1, cols, figsize=(6 * cols, 5), sharey=False)
    axes3 = np.atleast_1d(axes3)
    for col_idx, (name, _) in enumerate(solver_defs):
        ax = axes3[col_idx]
        for pt_idx, param in enumerate(PARAM_SETS):
            pname = param["name"]
            vals  = conv_data[name].get(pname)
            if not vals:
                continue
            gaps = np.array([v - f_star for v in vals])
            pos  = gaps > 0
            if not pos.any():
                continue
            color = colors[pt_idx % len(colors)]
            ks = np.arange(1, len(gaps) + 1)
            ax.loglog(ks[pos], gaps[pos], "-o", ms=4, lw=2, color=color, label=pname)

        # Reference slopes anchored at first available trace
        first_vals = next(
            (v for v in conv_data[name].values() if v and any(vi - f_star > 0 for vi in v)),
            None,
        )
        if first_vals is not None and (first_vals[0] - f_star) > 0:
            gap0   = first_vals[0] - f_star
            ks_ref = np.logspace(0, np.log10(max(len(first_vals), 10)), 50)
            for exp, rcol, rlbl in RATE_REFS:
                ax.loglog(ks_ref, gap0 * ks_ref ** (-exp), "--", lw=1.5,
                          color=rcol, label=rlbl, alpha=0.7)
        _format_ax(ax, name, "iteration $k$ (log)", "$f(x_k) - f^*$ (log)")
    _show_or_close(fig3, f"Rate verification (log-log) — {FUNCTION_NAME}", SHOW_RATE)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\nBest final objective per solver:")
    for name, _ in solver_defs:
        best_f, best_x, best_p = np.inf, None, None
        for pname, x_final in final_data[name].items():
            if x_final is not None and f(x_final) < best_f:
                best_f, best_x, best_p = f(x_final), x_final, pname
        if best_x is not None:
            print(f"{name}: f={best_f:.6e}, param={best_p}, x={best_x}, grad={grad(best_x)}")
        else:
            print(f"{name}: no result")


if __name__ == "__main__":
    main()
