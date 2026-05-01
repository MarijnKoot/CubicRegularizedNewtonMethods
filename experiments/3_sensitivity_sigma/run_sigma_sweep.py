"""
run_sigma_sweep.py — Driver for the σ₀ grid-sweep experiment.

Usage (from repo root, with venv active):
    python experiments/3_sensitivity_sigma/run_sigma_sweep.py

Outputs under experiments/3_sensitivity_sigma/results/
    raw/raw_results.csv
    summary/aggregated.csv
    summary/latex_tables.txt
    figures/heatmap_{method}.png
    figures/convergence_{problem}.png
    markdown/summary.md
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
sys.path.insert(0, str(_HERE.parent))

from sigma_sweep_runner import (
    SigmaResult, run_all,
    METHODS, PROBLEMS, SIGMA_GRID, SEEDS, EPS_G, MAX_ITER,
)

# ── output dirs ───────────────────────────────────────────────────────────────
_RES    = _ROOT / "results" / "3_sensitivity_sigma"
RAW_DIR = _RES
SUM_DIR = _RES
FIG_DIR = _RES / "figures"
MD_DIR  = _RES
_RES.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")

def _success_rate(grp):
    return sum(1 for r in grp if r.success) / len(grp) if grp else float("nan")


# ── CSV output ────────────────────────────────────────────────────────────────

def write_raw_csv(results):
    path = RAW_DIR / "raw_results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SigmaResult.csv_fieldnames())
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())
    print(f"\nRaw CSV    → {path}  ({len(results)} rows)")


def write_aggregated_csv(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r.label, r.method, r.sigma0)].append(r)

    fieldnames = [
        "label", "method", "sigma0", "sigma_ref",
        "n_runs", "success_rate",
        "iters_median", "iters_max",
        "rejected_median", "runtime_median",
        "final_f_median", "final_grad_norm_median",
    ]
    rows = []
    for (lbl, meth, s0), grp in sorted(groups.items()):
        ok     = [r for r in grp if r.success]
        iters  = [r.iterations for r in ok if r.iterations > 0]
        rejs   = [r.rejected_iter for r in grp]
        rts    = [r.runtime_sec for r in grp if np.isfinite(r.runtime_sec)]
        fvals  = [r.final_f for r in ok if np.isfinite(r.final_f)]
        gnorms = [r.final_grad_norm for r in ok if np.isfinite(r.final_grad_norm)]
        srefs  = [r.sigma_ref for r in grp if np.isfinite(r.sigma_ref)]
        rows.append({
            "label":                  lbl,
            "method":                 meth,
            "sigma0":                 s0,
            "sigma_ref":              _med(srefs),
            "n_runs":                 len(grp),
            "success_rate":           round(_success_rate(grp), 3),
            "iters_median":           _med(iters),
            "iters_max":              max(iters) if iters else 0,
            "rejected_median":        _med(rejs),
            "runtime_median":         _med(rts),
            "final_f_median":         _med(fvals),
            "final_grad_norm_median": _med(gnorms),
        })

    path = SUM_DIR / "aggregated.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Aggregated → {path}  ({len(rows)} rows)")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def write_latex_tables(results):
    """
    One table per method (NCR, ARC).
    Rows = problems, columns = 5 representative σ₀ values from the grid.
    Cell = success rate / median iterations.
    Grid style with |hline|.
    """
    groups = defaultdict(list)
    for r in results:
        groups[(r.label, r.method, r.sigma0)].append(r)

    prob_labels = [lbl for _, _, _, lbl in PROBLEMS]

    # Pick 5 representative σ₀ columns: indices 0, 3, 5 (≈mid), 8, 11
    col_indices = [0, 3, 5, 8, 11]
    col_sigmas  = [SIGMA_GRID[i] for i in col_indices]
    col_headers = [f"$10^{{{np.log10(s):.1f}}}$" for s in col_sigmas]

    # σ_ref per problem (median across all results for that label)
    sigma_refs = {}
    for _, _, _, lbl in PROBLEMS:
        refs = [r.sigma_ref for r in results if r.label == lbl and np.isfinite(r.sigma_ref)]
        sigma_refs[lbl] = _med(refs)

    n_runs = len(SEEDS)
    lines = []

    for method in [m for m in METHODS if m != "Newton"]:
        col_spec = "|l|" + "|".join("c" for _ in col_sigmas) + "|"

        lines.append(f"\n% ── {method}: σ₀ sensitivity ──────────────────────────────")
        lines.append(r"\begin{table}[ht]")
        lines.append(r"  \centering")
        caption = (
            f"$\\sigma_0$ sensitivity for \\textbf{{{method}}} "
            f"({n_runs} seeds per cell, $\\varepsilon_g = 10^{{-7}}$, $n=10$ unless noted). "
            r"Cell: \textit{rate} / \textit{med.\ $k$}. "
            r"Rate 1.00 = all seeds converged. "
            r"$\dagger$ marks the column closest to $\sigma_\mathrm{ref} = 2L_3$."
        )
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{tab:sigma_{method.lower()}}}")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \hline")

        # header row — mark column closest to σ_ref with †
        # (use a global average σ_ref across all problems)
        all_srefs = [v for v in sigma_refs.values() if np.isfinite(v) and v > 0]
        global_sref = float(np.median(all_srefs)) if all_srefs else float("nan")
        if np.isfinite(global_sref) and global_sref > 0:
            ref_dists = [abs(np.log10(max(s, 1e-20)) - np.log10(global_sref)) for s in col_sigmas]
            ref_col_idx = int(np.argmin(ref_dists))
        else:
            ref_col_idx = -1

        header_cells = []
        for ci, hdr in enumerate(col_headers):
            header_cells.append(f"{hdr}$^\\dagger$" if ci == ref_col_idx else hdr)
        lines.append("    \\textbf{Problem} & " + " & ".join(header_cells) + r" \\")
        lines.append(r"    \hline\hline")

        for lbl in prob_labels:
            cells = []
            for s0 in col_sigmas:
                grp = groups.get((lbl, method, s0), [])
                if not grp:
                    cells.append("---")
                else:
                    rate = _success_rate(grp)
                    ok_iters = [r.iterations for r in grp if r.success and r.iterations > 0]
                    med_k = _med(ok_iters)
                    k_str = f"{med_k:.0f}" if np.isfinite(med_k) else "---"
                    cells.append(f"{rate:.2f} / {k_str}")
            lines.append(f"    {lbl} & " + " & ".join(cells) + r" \\")
            lines.append(r"    \hline")

        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

    path = SUM_DIR / "latex_tables.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX      → {path}")
    return path


# ── Markdown summary ──────────────────────────────────────────────────────────

def write_markdown(results):
    from datetime import datetime
    groups = defaultdict(list)
    for r in results:
        groups[(r.label, r.method, r.sigma0)].append(r)

    lines = [
        "# σ₀ Sensitivity Experiment",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Goal",
        "",
        "Sweep the initial regularisation parameter σ₀ over a wide log-spaced grid",
        f"({len(SIGMA_GRID)} values from {SIGMA_GRID[0]:.0e} to {SIGMA_GRID[-1]:.0e})",
        "and measure how sensitive each method's convergence is to this choice.",
        "",
        "## Protocol",
        "",
        f"- Methods: {METHODS}",
        f"- Problems: {[lbl for _,_,_,lbl in PROBLEMS]}",
        f"- Seeds: {SEEDS[0]}..{SEEDS[-1]}  (N={len(SEEDS)} per cell)",
        f"- eps_g = {EPS_G},  max_iter = {MAX_ITER}",
        "- Newton is σ-independent; shown as a baseline column.",
        "",
        "## σ_ref",
        "",
        "The theoretical reference σ_ref = 2·L3 is marked with † in tables and",
        "with a dashed blue line in the heatmaps.  L3 is exact where known",
        "(quadratic: L3=0; cubic_norm: L3=2) and estimated via finite differences otherwise.",
        "",
        "## Results by Problem",
        "",
    ]

    prob_labels = [lbl for _, _, _, lbl in PROBLEMS]
    for lbl in prob_labels:
        lines.append(f"### {lbl}")
        lines.append("")
        lines.append("| Method | σ₀ | Rate | Med. k |")
        lines.append("|--------|-----|------|--------|")
        for method in METHODS:
            sigma_iter = [float("nan")] if method == "Newton" else SIGMA_GRID
            for s0 in sigma_iter:
                grp = groups.get((lbl, method, s0), [])
                if not grp:
                    continue
                rate = _success_rate(grp)
                ok_iters = [r.iterations for r in grp if r.success and r.iterations > 0]
                med_k = _med(ok_iters)
                s0_str = "N/A" if not np.isfinite(s0) else f"{s0:.2e}"
                k_str  = f"{med_k:.0f}" if np.isfinite(med_k) else "---"
                lines.append(f"| {method} | {s0_str} | {rate:.2f} | {k_str} |")
        lines.append("")

    lines += [
        "## Output Files",
        "",
        "| Path | Description |",
        "|------|-------------|",
        "| `raw/raw_results.csv` | One row per (problem, method, σ₀, seed) |",
        "| `summary/aggregated.csv` | Grouped by (problem, method, σ₀) |",
        "| `summary/latex_tables.txt` | LaTeX grid tables |",
        "| `figures/heatmap_{method}.png` | Success rate + iteration heatmaps |",
        "| `figures/convergence_{problem}.png` | Convergence curves by σ₀ |",
        "",
    ]

    path = MD_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown   → {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("σ₀ grid sweep experiment")
    print(f"Methods  : {METHODS}")
    print(f"σ₀ grid  : {SIGMA_GRID[0]:.1e} … {SIGMA_GRID[-1]:.1e}  ({len(SIGMA_GRID)} values)")
    print(f"Problems : {[lbl for _,_,_,lbl in PROBLEMS]}")
    print(f"Seeds    : {SEEDS[0]}…{SEEDS[-1]}  (N={len(SEEDS)})")
    print(f"eps_g    : {EPS_G}   max_iter : {MAX_ITER}")
    total = len(PROBLEMS) * (len(SIGMA_GRID) * 2 + 1) * len(SEEDS)
    print(f"Total runs: ~{total}")
    print("=" * 60)

    results = run_all(verbose=True)

    write_raw_csv(results)
    write_aggregated_csv(results)
    write_latex_tables(results)

    from plot_sigma_sweep import plot_all
    plot_all(results, SIGMA_GRID, PROBLEMS, METHODS)

    write_markdown(results)

    print("\n\n" + "=" * 60)
    print("LaTeX tables")
    print("=" * 60)
    print((SUM_DIR / "latex_tables.txt").read_text(encoding="utf-8"))

    print("\nDone.")


if __name__ == "__main__":
    main()
