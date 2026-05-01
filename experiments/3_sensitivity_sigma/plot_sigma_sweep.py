"""
plot_sigma_sweep.py — Heatmaps and convergence curves for the σ₀ grid sweep.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_HERE   = Path(__file__).resolve()
_RES    = _HERE.parents[2] / "results" / "3_sensitivity_sigma"
FIG_DIR = _RES / "figures"


def _med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _success_rate(grp):
    return sum(1 for r in grp if r.success) / len(grp) if grp else float("nan")


# ── heatmaps: success rate and median iterations ───────────────────────────────

def plot_heatmaps(results, sigma_grid, problems, methods):
    """
    Per method: one combined side-by-side PDF (rate | iters) with shared x-axis
    labels and a single colorbar per panel, plus individual PDFs for each metric.
    """
    groups = defaultdict(list)
    for r in results:
        groups[(r.label, r.method, r.sigma0)].append(r)

    sigma_refs = {}
    for prob_name, n, prob_kwargs, label in problems:
        refs = [r.sigma_ref for r in results if r.label == label and np.isfinite(r.sigma_ref)]
        sigma_refs[label] = float(np.median(refs)) if refs else float("nan")

    prob_labels = [lbl for _, _, _, lbl in problems]
    sigma_labels = [f"{s:.1e}" for s in sigma_grid]
    n_rows = len(prob_labels)
    n_cols = len(sigma_grid)

    def _annotate(ax, mat, fmt, norm, n_rows, n_cols):
        for i in range(n_rows):
            for j in range(n_cols):
                v = mat[i, j]
                if np.isfinite(v):
                    norm_v = float(norm(v)) if norm is not None else v
                    color = "white" if norm_v < 0.35 or norm_v > 0.80 else "black"
                    ax.text(j, i, fmt(v), ha="center", va="center",
                            fontsize=6.5, color=color, fontweight="bold")

    def _ref_borders(ax, ref_col):
        for i, j in ref_col.items():
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=2.5, edgecolor="royalblue", facecolor="none", zorder=5,
            ))

    def _save(fig, fname):
        out = FIG_DIR / fname
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure → {out}")

    cell = 0.62
    h_fig = n_rows * cell + 2.0

    for method in [m for m in methods if m != "Newton"]:
        rate_mat  = np.full((n_rows, n_cols), float("nan"))
        iters_mat = np.full((n_rows, n_cols), float("nan"))

        for i, lbl in enumerate(prob_labels):
            for j, s0 in enumerate(sigma_grid):
                grp = groups.get((lbl, method, s0), [])
                if grp:
                    rate_mat[i, j] = _success_rate(grp)
                    all_iters = [r.iterations for r in grp if r.iterations > 0]
                    iters_mat[i, j] = _med(all_iters)

        # σ_ref column per row
        ref_col = {}
        for i, lbl in enumerate(prob_labels):
            sref = sigma_refs.get(lbl, float("nan"))
            if np.isfinite(sref) and sref > 0:
                dists = [abs(np.log10(max(s, 1e-20)) - np.log10(sref)) for s in sigma_grid]
                ref_col[i] = int(np.argmin(dists))

        norm_rate = mcolors.Normalize(vmin=0, vmax=1)

        iter_plot = np.where(np.isfinite(iters_mat) & (iters_mat > 0), iters_mat, np.nan)
        valid = iter_plot[np.isfinite(iter_plot)]
        if len(valid) > 0:
            norm_iter = mcolors.LogNorm(vmin=max(valid.min(), 1),
                                        vmax=max(valid.max(), 2))
        else:
            norm_iter = None

        # ── individual PDFs ───────────────────────────────────────────────────
        for mat, norm, fmt, cbar_lbl, title, fname in [
            (rate_mat,  norm_rate, lambda v: f"{v:.2f}",
             "Rate", f"{method} — Success rate  (blue = $\\sigma_{{\\rm ref}}$)",
             f"heatmap_{method}_rate.pdf"),
            (iter_plot, norm_iter, lambda v: f"{v:.0f}",
             "Med. iterations", f"{method} — Median iterations  (blue = $\\sigma_{{\\rm ref}}$)",
             f"heatmap_{method}_iters.pdf"),
        ]:
            w_fig = n_cols * cell + 3.8
            fig, ax = plt.subplots(figsize=(w_fig, h_fig))
            im = ax.imshow(mat, cmap="RdYlGn" if "rate" in fname else "YlOrRd",
                           norm=norm, aspect="equal")
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(sigma_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(prob_labels, fontsize=9)
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel("$\\sigma_0$", fontsize=10)
            plt.colorbar(im, ax=ax, label=cbar_lbl, shrink=0.6, pad=0.02)
            _annotate(ax, mat, fmt, norm, n_rows, n_cols)
            _ref_borders(ax, ref_col)
            fig.tight_layout()
            _save(fig, fname)

        # ── combined side-by-side PDF ─────────────────────────────────────────
        # No colorbars; numbers in cells carry the information.
        # Panels are close together; y-labels (problem names) on left only;
        # metric label as ylabel on each panel.
        w_combined = 2 * n_cols * cell + 4.5
        fig, axes = plt.subplots(1, 2, figsize=(w_combined, h_fig),
                                 gridspec_kw={"wspace": 0.08})

        for ax, mat, norm, fmt, cmap, ylabel, panel_title in [
            (axes[0], rate_mat,  norm_rate, lambda v: f"{v:.2f}",
             "RdYlGn", "Rate", "Success rate"),
            (axes[1], iter_plot, norm_iter, lambda v: f"{v:.0f}",
             "YlOrRd", "Med. iterations", "Median iterations"),
        ]:
            ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(sigma_labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("$\\sigma_0$", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(panel_title, fontsize=11, pad=6)
            _annotate(ax, mat, fmt, norm, n_rows, n_cols)
            _ref_borders(ax, ref_col)

        # problem-name y-tick labels on left panel only
        axes[0].set_yticks(range(n_rows))
        axes[0].set_yticklabels(prob_labels, fontsize=9)
        axes[1].set_yticks(range(n_rows))
        axes[1].set_yticklabels([])

        fig.suptitle(f"{method}  —  $\\sigma_0$ sensitivity  "
                     f"(blue border = $\\sigma_{{\\rm ref}}$)", fontsize=12, y=1.01)
        _save(fig, f"heatmap_{method}.pdf")



# ── convergence curves ────────────────────────────────────────────────────────

def plot_convergence_curves(results, sigma_grid, problems, methods):
    """
    For each problem: f(x_k) vs iteration for 3 representative σ₀ values,
    one subplot per method (NCR, ARC).  Uses the solver log collected via
    the runner — we store only aggregated stats in SigmaResult, so this plot
    re-runs the solver for the representative σ₀ values.
    """
    # We stored only aggregated results; re-run 3 σ₀ per problem for curve plotting.
    import sys
    _root = _HERE.parents[2]
    for _p in [_root, _root/"src", _root/"experiments"]:
        s = str(_p)
        if s not in sys.path:
            sys.path.insert(0, s)

    from harness.problems  import get_problem
    from harness.counters  import EvalCounter
    from utilities         import estimate_L3
    from methods.NCR       import CubicNewton, CRNOptions
    from methods.ARC       import AdaptiveCubicNewton, ARCParams
    from methods.ACRN      import AcceleratedCubicNewton

    SEED = 42
    EPS_G_CURVE = 1e-7
    MAX_ITER_CURVE = 500

    curve_methods = [m for m in methods if m != "Newton"]
    colors = {"small": "C3", "ref": "C2", "large": "C0"}
    styles = {"small": "--", "ref": "-", "large": ":"}

    for prob_name, n, prob_kwargs, label in problems:
        f_raw, grad_raw, hess_raw, meta = get_problem(prob_name, n, seed=SEED,
                                                       problem_kwargs=prob_kwargs)
        L3_known = meta.get("L3")
        if L3_known is not None and L3_known > 0:
            sigma_ref = 2.0 * float(L3_known)
        elif L3_known == 0.0:
            sigma_ref = max(2.0 * estimate_L3(hess_raw, np.full(n, 1.0)), 1e-4)
        else:
            try:
                x0tmp = np.full(n, -1.0) if prob_name == "rosenbrock" else np.ones(n)
                sigma_ref = 2.0 * float(estimate_L3(hess_raw, x0tmp))
            except Exception:
                sigma_ref = 1.0

        # pick 3 representative σ₀: near lower end, near σ_ref, near upper end
        sigma_small = sigma_grid[1]
        sigma_large = sigma_grid[-2]
        sigma_mid   = sigma_ref
        sigma_cases = {"small": sigma_small, "ref": sigma_mid, "large": sigma_large}

        if prob_name == "rosenbrock":
            x0 = np.full(n, -1.0)
        else:
            x0 = np.random.default_rng(SEED + 1000).standard_normal(n)

        fig, axes = plt.subplots(1, len(curve_methods),
                                 figsize=(5 * len(curve_methods), 4), squeeze=False)

        for ax, method in zip(axes[0], curve_methods):
            for case_name, s0 in sigma_cases.items():
                s = max(float(s0), 1e-15)
                try:
                    counter = EvalCounter(f_raw, grad_raw, hess_raw)
                    if method == "NCR":
                        opts = CRNOptions(sigma0=s, sigma_min=1e-15, sigma_max=5e11,
                                          tol_grad=EPS_G_CURVE, max_iter=MAX_ITER_CURVE)
                        solver = CubicNewton(counter.f, counter.grad, counter.hess, options=opts)
                        solver.run(x0.copy())
                    elif method == "ARC":
                        params = ARCParams(sigma0=s, sigma_min=1e-15, sigma_max=5e11,
                                           eta1=0.1, eta2=0.9,
                                           tol_grad=EPS_G_CURVE, max_iter=MAX_ITER_CURVE)
                        solver = AdaptiveCubicNewton(counter.f, counter.grad, counter.hess,
                                                     params=params, step_method="secular")
                        solver.run(x0.copy())
                    else:  # ACRN
                        L3_eff = max(sigma_ref / 2.0, 1e-15)
                        solver = AcceleratedCubicNewton(
                            counter.f, counter.grad, counter.hess,
                            L3=L3_eff, sigma=s,
                            sigma_min=1e-15, sigma_max=5e11,
                            tol_grad=EPS_G_CURVE, max_iter=MAX_ITER_CURVE,
                            adaptive_sigma=True,
                        )
                        solver.run(x0.copy())

                    f_vals = [float(e["f"]) for e in solver.log]
                    if not f_vals:
                        continue
                    lbl_c = f"$\\sigma_0$={s0:.1e}" + (" (ref)" if case_name == "ref" else "")
                    if all(v > 0 for v in f_vals):
                        ax.semilogy(range(len(f_vals)), f_vals,
                                    color=colors[case_name], linestyle=styles[case_name],
                                    linewidth=1.8, label=lbl_c)
                    else:
                        ax.plot(range(len(f_vals)), f_vals,
                                color=colors[case_name], linestyle=styles[case_name],
                                linewidth=1.8, label=lbl_c)
                except Exception:
                    pass

            ax.set_title(f"{method}", fontsize=10)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("$f(x_k)$")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{label} — convergence by $\\sigma_0$", fontsize=11)
        fig.tight_layout()
        safe_label = label.replace(" ", "_").replace("=", "").replace("κ", "kappa")
        out = FIG_DIR / f"convergence_{safe_label}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure → {out}")


# ── main dispatcher ────────────────────────────────────────────────────────────

def plot_all(results, sigma_grid, problems, methods):
    print("\n--- Generating plots ---")
    plot_heatmaps(results, sigma_grid, problems, methods)
    plot_convergence_curves(results, sigma_grid, problems, methods)
