"""
run_experiment.py — Stopping criteria sensitivity experiment.

Runs each (problem, method) pair over N_RUNS seeds and reports
success rates + median iterations as LaTeX tables printed to stdout.

One table per stopping mode, rows = method, columns = (problem x eps_g).

Usage (from repo root, with venv active):
    python experiments/0_stopping_criteria_sensitivity/run_experiment.py

CSV output saved to:
    results/stopping_sensitivity/raw_results.csv
    results/stopping_sensitivity/aggregated.csv
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

from stopping_runner import (
    StoppingResult, run_problem_method,
    METHODS, EPS_GRID, MODES, EPS_X, EPS_F,
)

# ── configuration ─────────────────────────────────────────────────────────────
N_RUNS = 10                              # number of seeds per (problem, method)
SEEDS  = list(range(42, 42 + N_RUNS))   # seeds 42..51

PROBLEMS = [
    ("cubic_norm", 10),
    ("logsumexp",  10),
    ("rosenbrock", 10),
]

OUT_DIR = _ROOT / "results" / "0_stopping_criteria_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROB_LABELS = {
    "cubic_norm": "CubicNorm",
    "logsumexp":  "LogSumExp",
    "rosenbrock": "Rosenbrock",
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stopping criteria sensitivity  (LaTeX tables)")
    print(f"Problems : {[p for p,_ in PROBLEMS]}")
    print(f"Methods  : {METHODS}")
    print(f"eps_g    : {EPS_GRID}")
    print(f"Modes    : {MODES}")
    print(f"Seeds    : {SEEDS[0]}..{SEEDS[-1]}  (N={N_RUNS})")
    print("=" * 60)

    all_results: list[StoppingResult] = []

    for prob_name, n in PROBLEMS:
        print(f"\n--- {prob_name} (n={n}) ---")
        for method in METHODS:
            for seed in SEEDS:
                results = run_problem_method(prob_name, n, method,
                                             verbose=False, seed=seed)
                all_results.extend(results)
            print(f"  {method}: {N_RUNS} seeds done")

    # ── CSV outputs ───────────────────────────────────────────────────────────
    _write_raw_csv(all_results, OUT_DIR / "raw_results.csv")
    _write_aggregated_csv(all_results, OUT_DIR / "aggregated.csv")

    # ── termination reasons plot ──────────────────────────────────────────────
    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(exist_ok=True)
    _plot_termination_reasons(all_results, FIG_DIR)

    # ── LaTeX tables to stdout ────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("LaTeX tables  (copy into thesis)")
    print("=" * 60)

    for mode in MODES:
        print(f"\n% ── mode: {mode} ─────────────────────────────────────")
        print(_latex_table(all_results, mode))


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_raw_csv(results: list[StoppingResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=StoppingResult.csv_fieldnames())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())
    print(f"\nRaw CSV    → {path}  ({len(results)} rows)")


def _write_aggregated_csv(results: list[StoppingResult], path: Path) -> None:
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r.problem, r.method, r.stopping_mode, r.eps_g)].append(r)

    fieldnames = [
        "problem", "method", "stopping_mode", "eps_g",
        "n_runs", "success_rate",
        "iters_median", "iters_mean", "iters_min", "iters_max",
        "final_grad_norm_median",
    ]
    rows = []
    for (prob, meth, mode, epsg), grp in sorted(groups.items()):
        iters  = [r.iterations      for r in grp if r.iterations > 0]
        gnorms = [r.final_grad_norm for r in grp if np.isfinite(r.final_grad_norm)]
        rows.append({
            "problem":              prob,
            "method":               meth,
            "stopping_mode":        mode,
            "eps_g":                epsg,
            "n_runs":               len(grp),
            "success_rate":         float(np.mean([r.success for r in grp])),
            "iters_median":         float(np.median(iters))   if iters  else float("nan"),
            "iters_mean":           float(np.mean(iters))     if iters  else float("nan"),
            "iters_min":            int(np.min(iters))        if iters  else 0,
            "iters_max":            int(np.max(iters))        if iters  else 0,
            "final_grad_norm_median": float(np.median(gnorms)) if gnorms else float("nan"),
        })

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated → {path}  ({len(rows)} rows)")


# ── LaTeX table builder ───────────────────────────────────────────────────────

def _latex_table(results: list[StoppingResult], mode: str) -> str:
    """
    Build a single LaTeX booktabs table combining success rate and median
    iterations. Each cell shows  'rate / iters'  (e.g. '1.00 / 12').

    Rows    : one group per problem, one sub-row per eps_g value
    Columns : methods
    """
    # Aggregate both metrics
    groups: dict = defaultdict(list)
    for r in results:
        if r.stopping_mode == mode:
            groups[(r.problem, r.method, r.eps_g)].append(r)

    sr:  dict = {}
    med: dict = {}
    for (prob, meth, epsg), grp in groups.items():
        sr[(prob, meth, epsg)]  = float(np.mean([r.success for r in grp]))
        iters = [r.iterations for r in grp if r.iterations > 0]
        med[(prob, meth, epsg)] = float(np.median(iters)) if iters else float("nan")

    problems  = [p for p, _ in PROBLEMS]
    eps_strs  = [f"$10^{{{int(np.log10(e))}}}$" for e in EPS_GRID]

    mode_label = "gradient-only" if mode == "grad_only" else "gradient + small-step"
    caption = (
        f"Success rate / median iterations under {mode_label} stopping "
        f"({N_RUNS} seeds, $n=10$). "
        r"Cell format: \textit{rate} / \textit{iters}. "
        "Rate 1.00 = all runs converged; `---' = not applicable."
    )
    label = f"tab:stopping_{mode}"

    # col spec: ll (problem + eps_g) + one c per method
    col_spec = "ll" + "".join(" c" for _ in METHODS)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    lines.append("    Problem & $\\varepsilon_g$ & " + " & ".join(METHODS) + r" \\")
    lines.append(r"    \midrule")

    for i, prob in enumerate(problems):
        if i > 0:
            lines.append(r"    \midrule")
        for j, (epsg, eps_str) in enumerate(zip(EPS_GRID, eps_strs)):
            prob_cell = f"\\multirow{{{len(EPS_GRID)}}}{{*}}{{{PROB_LABELS.get(prob, prob)}}}" if j == 0 else ""
            cells = []
            for meth in METHODS:
                rate  = sr.get((prob, meth, epsg),  float("nan"))
                iters = med.get((prob, meth, epsg), float("nan"))
                if not np.isfinite(rate):
                    cells.append("---")
                else:
                    iters_str = f"{iters:.0f}" if np.isfinite(iters) else "---"
                    cells.append(f"{rate:.2f} / {iters_str}")
            lines.append(f"    {prob_cell} & {eps_str} & " + " & ".join(cells) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ── termination reasons plot ──────────────────────────────────────────────────

def _plot_termination_reasons(results: list[StoppingResult], fig_dir: Path) -> None:
    """
    Stacked bar chart: x = methods, bars = termination reason counts,
    one subplot per problem.  One figure saved per stopping mode.
    """
    from collections import Counter

    color_map = {
        "grad_tol": "C2",
        "step_tol": "C0",
        "f_tol":    "C1",
        "max_iter": "C3",
    }
    reasons_order = ["grad_tol", "step_tol", "f_tol", "max_iter"]
    problems = [p for p, _ in PROBLEMS]

    for mode in MODES:
        mode_results = [r for r in results if r.stopping_mode == mode]
        reasons_present = [r for r in reasons_order
                           if any(x.termination_reason == r for x in mode_results)]

        fig, axes = plt.subplots(1, len(problems),
                                 figsize=(4 * len(problems), 4), sharey=False)
        if len(problems) == 1:
            axes = [axes]

        for ax, prob in zip(axes, problems):
            counts = {m: Counter(r.termination_reason for r in mode_results
                                 if r.problem == prob and r.method == m)
                      for m in METHODS}
            x       = np.arange(len(METHODS))
            bottoms = np.zeros(len(METHODS))
            for reason in reasons_present:
                heights = np.array([counts[m].get(reason, 0) for m in METHODS],
                                   dtype=float)
                ax.bar(x, heights, bottom=bottoms,
                       color=color_map.get(reason, "gray"),
                       label=reason, alpha=0.85)
                bottoms += heights

            ax.set_xticks(x)
            ax.set_xticklabels(METHODS, fontsize=9)
            ax.set_title(PROB_LABELS.get(prob, prob))
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            if ax == axes[-1]:
                ax.legend(fontsize=7, loc="upper right")

        mode_label = "grad-only" if mode == "grad_only" else "grad+step"
        fig.suptitle(f"Termination reasons — {mode_label} stopping", fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"termination_reasons_{mode}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Figure → {out}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
