"""
run_experiment.py — Initial point sensitivity experiment.

Usage (from repo root, with venv active):
    python experiments/1_initial_point_sensitivity/run_experiment.py

Outputs saved to:
    results/initial_point_sensitivity/
        raw_results.csv
        aggregated.csv
        summary.md
        figures/
            {problem}_iters_boxplot.png
            {problem}_runtime_boxplot.png
            {problem}_final_f_boxplot.png
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
sys.path.insert(0, str(_HERE.parent))

from initial_point_runner import (
    InitPointResult, run_all,
    METHODS, PROBLEMS, SEED, N_RANDOM, PERTURB, EPS_G, MAX_ITER, DIM,
)

# ── output paths ──────────────────────────────────────────────────────────────
OUT_DIR = _ROOT / "results" / "1_initial_point_sensitivity"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {"Newton": "C0", "NCR": "C1", "ARC": "C2", "ACRN": "C3"}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Initial point sensitivity experiment")
    print(f"Problems : {[p for p,_,_ in PROBLEMS]}")
    print(f"Methods  : {METHODS}")
    print(f"Starts   : standard + benign + {N_RANDOM} random (perturb={PERTURB})")
    print(f"eps_g    : {EPS_G}   max_iter : {MAX_ITER}   dim : {DIM}")
    print(f"Seed     : {SEED}")
    print("=" * 60)

    results = run_all(verbose=True)

    _write_raw_csv(results,        OUT_DIR / "raw_results.csv")
    _write_aggregated_csv(results, OUT_DIR / "aggregated.csv")

    for prob_name, prob_label, _ in PROBLEMS:
        prob_results = [r for r in results if r.problem == prob_name]
        _plot_boxplot(prob_results, prob_name, prob_label, "iterations",
                      "Iterations", FIG_DIR / f"{prob_name}_iters_boxplot.png")
        _plot_boxplot(prob_results, prob_name, prob_label, "runtime_sec",
                      "Runtime (s)", FIG_DIR / f"{prob_name}_runtime_boxplot.png")
        _plot_boxplot(prob_results, prob_name, prob_label, "final_f",
                      "Final f(x)", FIG_DIR / f"{prob_name}_final_f_boxplot.png")

    _write_markdown(results, OUT_DIR / "summary.md")

    # ── contour path plots (2D projection) ────────────────────────────────────
    for prob_name, prob_label, prob_kwargs in PROBLEMS:
        _plot_contour_paths(prob_name, prob_label, prob_kwargs)

    print("\nDone.")


# ── CSV output ────────────────────────────────────────────────────────────────

def _write_raw_csv(results: list[InitPointResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=InitPointResult.csv_fieldnames())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())
    print(f"\nRaw CSV    → {path}  ({len(results)} rows)")


def _write_aggregated_csv(results: list[InitPointResult], path: Path) -> None:
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.problem, r.method)].append(r)

    fieldnames = [
        "problem", "method",
        "n_starts", "success_rate",
        "iters_median", "iters_iqr",
        "runtime_median",
        "final_f_median",
        "final_grad_norm_median",
        "sensitivity_ratio",   # max(iters) / min(iters) over successful runs
    ]
    rows = []
    for (prob, meth), grp in sorted(groups.items()):
        successes = [r for r in grp if r.success]
        iters_all = [r.iterations for r in grp if r.iterations > 0]
        iters_ok  = [r.iterations for r in successes]
        rts       = [r.runtime_sec  for r in grp if np.isfinite(r.runtime_sec)]
        fvals     = [r.final_f      for r in successes if np.isfinite(r.final_f)]
        gnorms    = [r.final_grad_norm for r in successes if np.isfinite(r.final_grad_norm)]

        q75, q25  = (np.percentile(iters_all, [75, 25]) if iters_all
                     else (float("nan"), float("nan")))
        sens = (max(iters_ok) / min(iters_ok)
                if len(iters_ok) >= 2 and min(iters_ok) > 0
                else float("nan"))

        rows.append({
            "problem":               prob,
            "method":                meth,
            "n_starts":              len(grp),
            "success_rate":          len(successes) / len(grp),
            "iters_median":          float(np.median(iters_all)) if iters_all else float("nan"),
            "iters_iqr":             float(q75 - q25),
            "runtime_median":        float(np.median(rts))   if rts   else float("nan"),
            "final_f_median":        float(np.median(fvals)) if fvals else float("nan"),
            "final_grad_norm_median": float(np.median(gnorms)) if gnorms else float("nan"),
            "sensitivity_ratio":     round(sens, 2) if np.isfinite(sens) else float("nan"),
        })

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated → {path}  ({len(rows)} rows)")


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_boxplot(
    results: list[InitPointResult],
    prob_name: str,
    prob_label: str,
    field: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    data   = []
    labels = []
    colors = []
    for meth in METHODS:
        vals = [getattr(r, field) for r in results
                if r.method == meth and np.isfinite(getattr(r, field, float("nan")))]
        if vals:
            data.append(vals)
            labels.append(meth)
            colors.append(METHOD_COLORS[meth])

    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(SEED).uniform(-0.15, 0.15, len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=color, alpha=0.6, s=30, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{prob_label} — {ylabel}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Figure → {path}")


# ── markdown summary ──────────────────────────────────────────────────────────

def _write_markdown(results: list[InitPointResult], path: Path) -> None:
    from datetime import datetime
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.problem, r.method)].append(r)

    lines = [
        "# Initial Point Sensitivity",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Problems",
        "",
    ]
    prob_desc = {
        "logsumexp":      f"LogSumExp (n={DIM}) — convex, random (a,b) with seed {SEED}",
        "quartic_convex": f"QuarticConvex (n={DIM}) — strongly convex, f(x)=0.5||x||²+0.25||x||⁴",
        "rosenbrock":     f"Rosenbrock (n={DIM}) — nonconvex, global min at x=(1,...,1)",
    }
    for prob_name, prob_label, _ in PROBLEMS:
        lines.append(f"- **{prob_label}**: {prob_desc.get(prob_name, prob_name)}")

    lines += [
        "",
        "## Starting Points",
        "",
        f"| Index | Label | Description |",
        f"|-------|-------|-------------|",
        f"| 0 | standard | Problem-specific fixed start (e.g. x=(-1,...,-1) for Rosenbrock) |",
        f"| 1 | benign | Closer to known minimizer |",
        f"| 2–{1+N_RANDOM} | random_i | standard + N(0, {PERTURB}²) noise, seed {SEED} |",
        "",
        "## Protocol",
        "",
        f"- `eps_g = {EPS_G}`, `max_iter = {MAX_ITER}`, `dim = {DIM}`, `seed = {SEED}`",
        "- ACRN skipped for nonconvex problems",
        "",
        "## Aggregated Results",
        "",
        "| Problem | Method | Success | Med. Iters | IQR | Sensitivity ratio |",
        "|---------|--------|---------|------------|-----|-------------------|",
    ]

    for (prob, meth), grp in sorted(groups.items()):
        successes = [r for r in grp if r.success]
        iters_all = [r.iterations for r in grp if r.iterations > 0]
        iters_ok  = [r.iterations for r in successes]
        q75, q25  = (np.percentile(iters_all, [75, 25]) if iters_all
                     else (float("nan"), float("nan")))
        sens = (f"{max(iters_ok)/min(iters_ok):.1f}"
                if len(iters_ok) >= 2 and min(iters_ok) > 0 else "—")
        med  = f"{np.median(iters_all):.0f}" if iters_all else "—"
        iqr  = f"{q75-q25:.0f}" if np.isfinite(q75) else "—"
        sr   = f"{len(successes)}/{len(grp)}"
        lines.append(f"| {prob} | {meth} | {sr} | {med} | {iqr} | {sens} |")

    lines += [
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `raw_results.csv` | One row per (problem, method, starting point) |",
        "| `aggregated.csv` | Summary stats grouped by (problem, method) |",
        "| `figures/{problem}_iters_boxplot.png` | Iteration count distribution |",
        "| `figures/{problem}_runtime_boxplot.png` | Runtime distribution |",
        "| `figures/{problem}_final_f_boxplot.png` | Final objective distribution |",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown   → {path}")


# ── contour path plots ────────────────────────────────────────────────────────

def _collect_paths(
    prob_name: str,
    prob_kwargs: dict,
    method: str,
    x0: np.ndarray,
) -> list[np.ndarray] | None:
    """
    Run the solver and return the list of x iterates (including x0).
    Returns None if the method is not applicable.
    """
    import sys
    sys.path.insert(0, str((_ROOT / "src").resolve()))

    from harness.problems  import get_problem
    from harness.counters  import EvalCounter
    from utilities         import estimate_L3
    from methods.pure_newton import PureNewton, PureNewtonOptions
    from methods.NCR         import CubicNewton, CRNOptions
    from methods.ARC         import AdaptiveCubicNewton, ARCParams
    from methods.ACRN        import AcceleratedCubicNewton

    f_raw, grad_raw, hess_raw, meta = get_problem(prob_name, len(x0), seed=SEED,
                                                   problem_kwargs=prob_kwargs)
    if method == "ACRN" and not meta.get("convex", False):
        return None

    counter = EvalCounter(f_raw, grad_raw, hess_raw)
    L3 = meta.get("L3")

    try:
        if method == "Newton":
            opts   = PureNewtonOptions(tol_grad=EPS_G, max_iter=MAX_ITER)
            solver = PureNewton(counter.f, counter.grad, counter.hess, options=opts)
            solver.run(x0.copy())
            log = solver.log
        elif method == "NCR":
            L3_pos = L3 if (L3 and L3 > 0) else 0.0
            opts   = CRNOptions(sigma0=max(2*L3_pos, 0.5), tol_grad=EPS_G, max_iter=MAX_ITER)
            solver = CubicNewton(counter.f, counter.grad, counter.hess, options=opts)
            solver.run(x0.copy())
            log = solver.log
        elif method == "ARC":
            params = ARCParams(tol_grad=EPS_G, max_iter=MAX_ITER)
            solver = AdaptiveCubicNewton(counter.f, counter.grad, counter.hess,
                                         params=params, step_method="secular")
            solver.run(x0.copy())
            log = solver.log
        elif method == "ACRN":
            L3_eff = L3 if (L3 and L3 > 0) else max(estimate_L3(hess_raw, x0), 1e-15)
            solver = AcceleratedCubicNewton(counter.f, counter.grad, counter.hess,
                                            L3=L3_eff, tol_grad=EPS_G, max_iter=MAX_ITER,
                                            adaptive_sigma=True)
            solver.run(x0.copy())
            log = solver.log
        else:
            return None
    except Exception:
        return None

    path = [x0.copy()] + [np.asarray(e["x"], dtype=float) for e in log]
    return path


def _plot_contour_paths(prob_name: str, prob_label: str, prob_kwargs: dict) -> None:
    """
    One figure per problem, one subplot per applicable method.
    Uses 3 starting points for clarity: standard, benign, one distant random.
    Grid bounds are derived from the actual iterate paths, not just the starts.
    Log-scale colormap is used for functions with large dynamic range.
    """
    from matplotlib.colors import LogNorm
    from harness.problems import get_problem
    from initial_point_runner import make_starting_points

    n = 2
    f_raw, _, _, meta = get_problem(prob_name, n, seed=SEED, problem_kwargs=prob_kwargs)

    # ── select 3 representative starts: standard, benign, most distant random ─
    all_starts = make_starting_points(prob_name, n)
    standard = all_starts[0]
    benign   = all_starts[1]
    randoms  = all_starts[2:]
    # pick the random start with the largest distance from standard
    distant  = max(randoms, key=lambda lx: np.linalg.norm(lx[1] - standard[1]))
    plot_starts = [standard, benign, distant]

    # ── collect all paths first so grid can cover them ────────────────────────
    applicable = [m for m in METHODS
                  if not (m == "ACRN" and not meta.get("convex", False))]

    all_paths = {}   # method -> list of paths (one per start)
    for method in applicable:
        all_paths[method] = []
        for label, x0 in plot_starts:
            path = _collect_paths(prob_name, prob_kwargs, method, x0)
            all_paths[method].append((label, x0, path))

    # ── grid bounds: cover all finite iterates across all methods/starts ──────
    all_pts = []
    for method, path_list in all_paths.items():
        for _, x0, path in path_list:
            if path:
                all_pts.extend(path)
    all_pts = [p for p in all_pts if np.all(np.isfinite(p))]

    if not all_pts:
        return

    xs_all = [p[0] for p in all_pts]
    ys_all = [p[1] for p in all_pts]
    margin = max(1.0, (max(xs_all) - min(xs_all)) * 0.12,
                      (max(ys_all) - min(ys_all)) * 0.12)
    x_lo, x_hi = min(xs_all) - margin, max(xs_all) + margin
    y_lo, y_hi = min(ys_all) - margin, max(ys_all) + margin

    # ── contour grid ──────────────────────────────────────────────────────────
    gx = np.linspace(x_lo, x_hi, 300)
    gy = np.linspace(y_lo, y_hi, 300)
    GX, GY = np.meshgrid(gx, gy)
    Z = np.array([[f_raw(np.array([xi, yi])) for xi in gx] for yi in gy])

    # Use log-norm if dynamic range > 100 (e.g. Rosenbrock)
    z_pos = Z[Z > 0]
    use_log = (len(z_pos) > 0 and z_pos.max() / max(z_pos.min(), 1e-12) > 100)
    if use_log:
        z_min = max(z_pos.min(), 1e-6)
        z_max = z_pos.max()
        norm  = LogNorm(vmin=z_min, vmax=z_max)
        levels = np.logspace(np.log10(z_min), np.log10(z_max), 25)
        Z_plot = np.clip(Z, z_min, z_max)
    else:
        Z_clipped = np.clip(Z, np.nanpercentile(Z, 2), np.nanpercentile(Z, 98))
        norm   = None
        levels = 20
        Z_plot = Z_clipped

    # ── style: one color per starting point ───────────────────────────────────
    start_styles = {
        "standard": ("C0", "o"),
        "benign":   ("C1", "s"),
    }
    start_styles[distant[0]] = ("C2", "^")
    # human-readable display names
    display_names = {"standard": "standard", "benign": "benign", distant[0]: "distant"}

    # ── figure: 2 subplots per row ────────────────────────────────────────────
    n_meth  = len(applicable)
    ncols   = min(n_meth, 2)
    nrows   = (n_meth + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 5.0 * nrows),
                              squeeze=False)
    ax_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for method, ax in zip(applicable, ax_flat):
        ax.contourf(GX, GY, Z_plot, levels=levels,
                    cmap="RdYlGn_r", alpha=0.55, norm=norm)
        ax.contour(GX, GY, Z_plot, levels=levels,
                   colors="k", linewidths=0.3, alpha=0.35, norm=norm)

        legend_handles = []
        for label, x0, path in all_paths[method]:
            color, marker = start_styles.get(label, ("gray", "o"))
            dname = display_names.get(label, label)

            finite = [p for p in path if np.all(np.isfinite(p))] if path else []
            n_iters = len(finite) - 1 if len(finite) >= 2 else 0  # exclude x0

            if len(finite) >= 2:
                px = [p[0] for p in finite]
                py = [p[1] for p in finite]

                # full path line
                ax.plot(px, py, color=color, linewidth=1.4, alpha=0.85, zorder=3)

                # dot at every iterate so individual steps are visible
                ax.scatter(px[1:-1], py[1:-1], color=color, s=18, zorder=4, alpha=0.7)

                # arrow showing direction of travel on last segment
                ax.annotate("", xy=(px[-1], py[-1]), xytext=(px[-2], py[-2]),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                            zorder=5)

            # start marker (not added to ax.plot legend — we build it manually)
            ax.plot(x0[0], x0[1], marker=marker, color=color,
                    markersize=9, zorder=7, linestyle="None")

            # legend entry: marker + "standard (k=12)"
            import matplotlib.lines as mlines
            handle = mlines.Line2D([], [], color=color, marker=marker,
                                   linestyle="-", markersize=7, linewidth=1.4,
                                   label=f"{dname} (k={n_iters})")
            legend_handles.append(handle)

        ax.legend(handles=legend_handles, fontsize=8, loc="best",
                  framealpha=0.85, handlelength=2.2)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.tick_params(labelsize=8)

    # hide unused subplots
    for ax in ax_flat[len(applicable):]:
        ax.set_visible(False)

    fig.suptitle(f"{prob_label} — paths from different starts ($n=2$)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / f"{prob_name}_contour_paths.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {out}")


if __name__ == "__main__":
    main()
