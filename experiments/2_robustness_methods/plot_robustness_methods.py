"""
plot_robustness_methods.py — All plots and the Markdown summary for the robustness experiment.
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

_HERE   = Path(__file__).resolve()
_RES    = _HERE.parents[2] / "results" / "2_robustness_methods"
FIG_DIR = _RES / "figures"
MD_DIR  = _RES

METHOD_COLORS = {"Newton": "C0", "NCR": "C1", "ARC": "C2", "ACRN": "C3"}
METHOD_MARKERS = {"Newton": "o", "NCR": "s", "ARC": "^", "ACRN": "D"}

_FAMILIES = ["ill_conditioned", "rosenbrock", "dixon_price", "rastrigin"]
_FAM_LABELS = {
    "ill_conditioned": "Ill-conditioned quadratics",
    "rosenbrock":      "Rosenbrock",
    "dixon_price":     "Dixon-Price",
    "rastrigin":       "Rastrigin",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _skip(grp):
    return not grp or all(r.termination_reason == "skip:nonconvex" for r in grp)


def _med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _success_rate(grp):
    return sum(1 for r in grp if r.success) / len(grp) if grp else float("nan")


def _active_methods(results, family):
    """Methods that ran at least once (not all skipped) in this family."""
    return [m for m in ["Newton", "NCR", "ARC", "ACRN"]
            if any(r.method == m and r.family == family
                   and r.termination_reason != "skip:nonconvex"
                   for r in results)]


# ── 1. Ill-conditioned quadratics: metrics vs condition number ─────────────────

def _plot_ill_conditioned(results):
    ill = [r for r in results if r.family == "ill_conditioned"
           and r.problem == "quadratic"]
    if not ill:
        return

    # group by (method, label) → average over x0s
    groups = defaultdict(list)
    for r in ill:
        groups[(r.method, r.label)].append(r)

    # extract condition numbers from labels "Quadratic κ=1e04" etc.
    def _cond(lbl):
        return float(lbl.split("=")[1].replace("e", "e"))

    labels_sorted = sorted(set(r.label for r in ill), key=_cond)
    conds = [_cond(l) for l in labels_sorted]

    metrics = [
        ("iters",     "Iterations",          lambda grp: _med([r.iterations for r in grp if r.success])),
        ("runtime",   "Runtime (s)",          lambda grp: _med([r.runtime_sec for r in grp])),
        ("grad_norm", "Final $\\|\\nabla f\\|$", lambda grp: _med([r.final_grad_norm for r in grp if r.success])),
    ]

    active = _active_methods(results, "ill_conditioned")
    # filter to only quadratic methods
    active = [m for m in active if any(r.method == m and r.problem == "quadratic" for r in results)]

    for met_key, met_label, met_fn in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for meth in active:
            ys = [met_fn(groups.get((meth, lbl), [])) for lbl in labels_sorted]
            ax.plot(conds, ys, color=METHOD_COLORS[meth],
                    marker=METHOD_MARKERS[meth], label=meth, linewidth=1.8)
        ax.set_xscale("log")
        if met_key == "grad_norm":
            ax.set_yscale("log")
        ax.set_xlabel("Condition number $\\kappa$")
        ax.set_ylabel(met_label)
        ax.set_title(f"Ill-conditioned quadratic — {met_label} vs $\\kappa$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = FIG_DIR / f"ill_cond_{met_key}_vs_kappa.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  Figure → {out}")

    # success/failure heatmap-style bar
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(labels_sorted))
    width = 0.8 / len(active)
    for i, meth in enumerate(active):
        rates = [_success_rate(groups.get((meth, lbl), [])) for lbl in labels_sorted]
        ax.bar(x + (i - len(active)/2 + 0.5)*width, rates,
               width=width, color=METHOD_COLORS[meth], label=meth, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"κ={c:.0e}" for c in conds], fontsize=8)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Ill-conditioned quadratic — success rate vs $\\kappa$")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "ill_cond_success_vs_kappa.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Figure → {out}")


# ── 2. Rosenbrock-2: path plot ─────────────────────────────────────────────────

def _plot_rosenbrock2_paths(results):
    """Contour + path plot for Rosenbrock n=2 for each method."""
    import sys
    _root = _HERE.parents[2]
    for _p in [_root, _root/"src", _root/"experiments"]:
        s = str(_p)
        if s not in sys.path:
            sys.path.insert(0, s)

    from harness.problems  import get_problem
    from harness.counters  import EvalCounter
    from utilities         import estimate_L3
    from methods.pure_newton import PureNewton, PureNewtonOptions
    from methods.NCR         import CubicNewton, CRNOptions
    from methods.ARC         import AdaptiveCubicNewton, ARCParams
    from matplotlib.colors import LogNorm

    SEED = 42
    EPS_G = 1e-7
    MAX_ITER = 500
    n = 2
    f_raw, grad_raw, hess_raw, meta = get_problem("rosenbrock", n, seed=SEED)

    x0s = {"standard": np.array([-1.0, -1.0]), "benign": np.array([0.5, 0.5])}

    def _collect(method, x0):
        counter = EvalCounter(f_raw, grad_raw, hess_raw)
        try:
            if method == "Newton":
                opts = PureNewtonOptions(tol_grad=EPS_G, max_iter=MAX_ITER)
                solver = PureNewton(counter.f, counter.grad, counter.hess, options=opts)
                solver.run(x0.copy())
            elif method == "NCR":
                opts = CRNOptions(tol_grad=EPS_G, max_iter=MAX_ITER)
                solver = CubicNewton(counter.f, counter.grad, counter.hess, options=opts)
                solver.run(x0.copy())
            elif method == "ARC":
                params = ARCParams(tol_grad=EPS_G, max_iter=MAX_ITER)
                solver = AdaptiveCubicNewton(counter.f, counter.grad, counter.hess,
                                             params=params, step_method="secular")
                solver.run(x0.copy())
            else:
                return None
            path = [x0.copy()] + [np.asarray(e["x"], dtype=float) for e in solver.log]
            return path
        except Exception:
            return None

    active = ["Newton", "NCR", "ARC"]
    all_paths = {m: {lbl: _collect(m, x0) for lbl, x0 in x0s.items()} for m in active}

    # grid bounds
    all_pts = []
    for m, ps in all_paths.items():
        for lbl, path in ps.items():
            if path:
                all_pts.extend([p for p in path if np.all(np.isfinite(p))])
    if not all_pts:
        return
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    mg = max(0.5, (max(xs)-min(xs))*0.1, (max(ys)-min(ys))*0.1)
    x_lo, x_hi = min(xs)-mg, max(xs)+mg
    y_lo, y_hi = min(ys)-mg, max(ys)+mg

    gx = np.linspace(x_lo, x_hi, 300)
    gy = np.linspace(y_lo, y_hi, 300)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.array([[f_raw(np.array([xi, yi])) for xi in gx] for yi in gy])
    z_pos = Z[Z > 0]
    use_log = len(z_pos) > 0 and z_pos.max() / max(z_pos.min(), 1e-12) > 100
    if use_log:
        z_min, z_max = max(z_pos.min(), 1e-6), z_pos.max()
        cnorm = LogNorm(vmin=z_min, vmax=z_max)
        levels = np.logspace(np.log10(z_min), np.log10(z_max), 25)
        Z_plot = np.clip(Z, z_min, z_max)
    else:
        cnorm = None; levels = 20; Z_plot = Z

    start_styles = {"standard": ("C0", "o"), "benign": ("C1", "s")}

    fig, axes = plt.subplots(1, len(active), figsize=(5*len(active), 4.5), squeeze=False)
    for ax, method in zip(axes[0], active):
        ax.contourf(GX, GY, Z_plot, levels=levels, cmap="RdYlGn_r", alpha=0.55, norm=cnorm)
        ax.contour(GX, GY, Z_plot, levels=levels, colors="k", linewidths=0.3, alpha=0.3, norm=cnorm)
        handles = []
        for lbl, x0 in x0s.items():
            color, marker = start_styles.get(lbl, ("gray", "o"))
            path = all_paths[method].get(lbl)
            finite = [p for p in path if np.all(np.isfinite(p))] if path else []
            n_iters = len(finite) - 1 if len(finite) >= 2 else 0
            if len(finite) >= 2:
                px = [p[0] for p in finite]; py = [p[1] for p in finite]
                ax.plot(px, py, color=color, linewidth=1.4, alpha=0.85, zorder=3)
                ax.scatter(px[1:-1], py[1:-1], color=color, s=16, zorder=4, alpha=0.7)
                if len(px) >= 2:
                    ax.annotate("", xy=(px[-1], py[-1]), xytext=(px[-2], py[-2]),
                                arrowprops=dict(arrowstyle="->", color=color, lw=1.4), zorder=5)
            ax.plot(x0[0], x0[1], marker=marker, color=color, markersize=9, zorder=7, linestyle="None")
            import matplotlib.lines as mlines
            h = mlines.Line2D([], [], color=color, marker=marker, linestyle="-",
                              markersize=7, linewidth=1.4, label=f"{lbl} (k={n_iters})")
            handles.append(h)
        ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.85)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)

    fig.suptitle("Rosenbrock (n=2) — optimization paths", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "rosenbrock2_paths.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


# ── 3. Bar plots: iterations and runtime by method and dimension ───────────────

def _plot_bar_by_dim(results, family_key, metric_field, ylabel, filename, log_y=False):
    fam_results = [r for r in results if r.family == family_key]
    if not fam_results:
        return

    active = _active_methods(results, family_key)
    dims   = sorted(set(r.n for r in fam_results))

    groups = defaultdict(list)
    for r in fam_results:
        groups[(r.method, r.n)].append(r)

    x = np.arange(len(dims))
    width = 0.8 / len(active)

    fig, ax = plt.subplots(figsize=(max(5, len(dims)*2), 4))
    for i, meth in enumerate(active):
        ys = []
        for n in dims:
            grp = groups.get((meth, n), [])
            if metric_field == "iterations":
                vals = [r.iterations for r in grp if r.success and r.iterations > 0]
            elif metric_field == "runtime_sec":
                vals = [r.runtime_sec for r in grp if np.isfinite(r.runtime_sec)]
            elif metric_field == "final_f":
                vals = [r.final_f for r in grp if r.success and np.isfinite(r.final_f)]
            elif metric_field == "final_grad_norm":
                vals = [r.final_grad_norm for r in grp if r.success and np.isfinite(r.final_grad_norm)]
            else:
                vals = []
            ys.append(_med(vals) if vals else float("nan"))
        offset = (i - len(active)/2 + 0.5) * width
        bars = ax.bar(x + offset, ys, width=width,
                      color=METHOD_COLORS[meth], label=meth, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={d}" for d in dims])
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(f"{_FAM_LABELS.get(family_key, family_key)} — {ylabel}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Figure → {out}")


# ── 4. Rejected steps summary ─────────────────────────────────────────────────

def _plot_rejected_steps(results):
    groups = defaultdict(list)
    for r in results:
        if r.method in ("NCR", "ARC") and r.termination_reason != "skip:nonconvex":
            groups[(r.family, r.method)].append(r)

    if not groups:
        return

    families = sorted(set(k[0] for k in groups))
    methods  = ["NCR", "ARC"]
    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, meth in enumerate(methods):
        ys = [_med([r.rejected_iter for r in groups.get((fam, meth), [])])
              for fam in families]
        ax.bar(x + (i-0.5)*width, ys, width=width,
               color=METHOD_COLORS[meth], label=meth, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([_FAM_LABELS.get(f, f) for f in families], rotation=15, ha="right")
    ax.set_ylabel("Median rejected iterations")
    ax.set_title("Rejected steps by family (NCR and ARC)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "rejected_steps_by_family.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Figure → {out}")


# ── 5. Success rate overview heatmap ──────────────────────────────────────────

def _plot_success_heatmap(results):
    active_methods = [m for m in ["Newton", "NCR", "ARC", "ACRN"]
                      if any(r.method == m for r in results)]
    labels_all = []
    for fam in _FAMILIES:
        fam_labels = sorted(set(r.label for r in results if r.family == fam),
                            key=lambda l: next((r.n for r in results if r.label == l), 0))
        labels_all.extend(fam_labels)

    if not labels_all:
        return

    groups = defaultdict(list)
    for r in results:
        if r.termination_reason != "skip:nonconvex":
            groups[(r.label, r.method)].append(r)

    matrix = np.full((len(labels_all), len(active_methods)), float("nan"))
    for i, lbl in enumerate(labels_all):
        for j, meth in enumerate(active_methods):
            grp = groups.get((lbl, meth), [])
            if grp:
                matrix[i, j] = _success_rate(grp)

    fig, ax = plt.subplots(figsize=(max(5, len(active_methods)*1.5),
                                    max(4, len(labels_all)*0.45)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(active_methods)))
    ax.set_xticklabels(active_methods, fontsize=9)
    ax.set_yticks(range(len(labels_all)))
    ax.set_yticklabels(labels_all, fontsize=7)
    ax.set_title("Success rate overview (green=1.0, red=0.0)", fontsize=10)
    plt.colorbar(im, ax=ax, label="Success rate")

    # annotate cells
    for i in range(len(labels_all)):
        for j in range(len(active_methods)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if 0.3 < v < 0.85 else "white")

    fig.tight_layout()
    out = FIG_DIR / "success_rate_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


# ── main plot dispatcher ───────────────────────────────────────────────────────

def plot_all(results):
    print("\n--- Generating plots ---")
    _plot_ill_conditioned(results)
    _plot_rosenbrock2_paths(results)

    for fam, fam_label in [("rosenbrock", "rosenbrock"),
                            ("dixon_price", "dixon_price"),
                            ("rastrigin", "rastrigin")]:
        _plot_bar_by_dim(results, fam, "iterations",  "Median iterations",
                         f"{fam}_iters_by_dim.png")
        _plot_bar_by_dim(results, fam, "runtime_sec", "Median runtime (s)",
                         f"{fam}_runtime_by_dim.png")
        _plot_bar_by_dim(results, fam, "final_f",     "Median final $f(x)$",
                         f"{fam}_final_f_by_dim.png", log_y=True)
        _plot_bar_by_dim(results, fam, "final_grad_norm", "Median $\\|\\nabla f\\|$",
                         f"{fam}_grad_norm_by_dim.png", log_y=True)

    _plot_rejected_steps(results)
    _plot_success_heatmap(results)


# ── Markdown summary ──────────────────────────────────────────────────────────

def write_markdown(results):
    from collections import defaultdict
    from datetime import datetime

    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.label, r.method)].append(r)

    lines = [
        "# Robustness of Methods w.r.t. Problem Geometry",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Definition of Robustness",
        "",
        "A method is considered **robust** if it consistently produces reliable, stable, and",
        "high-quality solutions when the problem geometry becomes difficult.",
        "Concretely, robustness is measured by:",
        "- **Success rate**: fraction of runs that satisfy ‖∇f‖ ≤ ε_g = 1e-7",
        "- **Iteration count**: median iterations over the set of starting points",
        "- **Numerical stability**: absence of NaN/Inf in iterates",
        "- **Sensitivity ratio**: max/min iterations across starts (lower = more robust)",
        "",
        "## Benchmark Families",
        "",
        "| Family | Problems | Geometric difficulty |",
        "|--------|----------|----------------------|",
        "| Ill-conditioned quadratics | SPD quadratics with κ ∈ {1e2, 1e4, 1e6, 1e8} + near-singular | Numerical stability, Newton breakdown |",
        "| Rosenbrock | n = 2, 10, 20 | Narrow curved valley, slow convergence |",
        "| Dixon-Price | n = 10, 20, 50 | Coupled variable interactions |",
        "| Rastrigin | n = 10, 20, 50 | Highly multimodal, many local minima |",
        "",
        "## Protocol",
        "",
        "- `eps_g = 1e-7` (gradient-norm stopping), `max_iter = 500`",
        "- 2 starting points per instance (standard + one variant)",
        "- ACRN only applied to convex problems",
        "- Seed = 42 throughout",
        "",
        "## Results Summary",
        "",
    ]

    for fam in _FAMILIES:
        fam_res = [r for r in results if r.family == fam]
        if not fam_res:
            continue
        lines.append(f"### {_FAM_LABELS.get(fam, fam)}")
        lines.append("")
        lines.append("| Problem | Method | Success | Med. iters | Med. runtime (s) | Rejected (med) |")
        lines.append("|---------|--------|---------|------------|------------------|----------------|")
        lbl_sorted = sorted(set(r.label for r in fam_res),
                            key=lambda l: next((r.n for r in fam_res if r.label == l), 0))
        for lbl in lbl_sorted:
            for meth in ["Newton", "NCR", "ARC", "ACRN"]:
                grp = groups.get((fam, lbl, meth), [])
                if not grp or all(r.termination_reason == "skip:nonconvex" for r in grp):
                    continue
                ok = [r for r in grp if r.success]
                iters = [r.iterations for r in ok if r.iterations > 0]
                rts   = [r.runtime_sec for r in grp if np.isfinite(r.runtime_sec)]
                rejs  = [r.rejected_iter for r in grp]
                sr    = f"{len(ok)}/{len(grp)}"
                med_i = f"{np.median(iters):.0f}" if iters else "---"
                med_r = f"{np.median(rts):.4f}" if rts else "---"
                med_j = f"{np.median(rejs):.0f}" if rejs else "0"
                lines.append(f"| {lbl} | {meth} | {sr} | {med_i} | {med_r} | {med_j} |")
        lines.append("")

    lines += [
        "## Output Files",
        "",
        "| Path | Description |",
        "|------|-------------|",
        "| `raw/raw_results.csv` | One row per (instance, method, start) |",
        "| `summary/aggregated.csv` | Grouped statistics |",
        "| `summary/latex_tables.txt` | LaTeX booktabs tables per family |",
        "| `figures/ill_cond_*` | Ill-conditioned quadratic plots |",
        "| `figures/rosenbrock2_paths.png` | Path plots for Rosenbrock n=2 |",
        "| `figures/{family}_*_by_dim.png` | Bar plots by dimension |",
        "| `figures/rejected_steps_by_family.png` | Rejected steps summary |",
        "| `figures/success_rate_heatmap.png` | Overview heatmap |",
        "",
    ]

    path = MD_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown  → {path}")
