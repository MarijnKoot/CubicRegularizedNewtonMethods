"""
run_synthesis.py — Large-scale synthesis experiment driver.

Usage (from repo root, venv active):
    python experiments/5_large_scale_synthesis/run_synthesis.py

Outputs under experiments/5_large_scale_synthesis/results/
    raw/raw_results.csv
    summary/aggregated.csv
    latex/table_{family}.tex
    figures/scaling_{family}.pdf
    markdown/summary.md
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy.linalg import norm

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_SRC  = _ROOT / "src"
for _p in (_ROOT, _SRC):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)
sys.path.insert(0, str(_HERE.parent))

from harness.problems  import get_problem
from harness.counters  import EvalCounter
from harness.adapters  import run_and_extract
from utilities         import estimate_L3
from benchmarks        import ALL_INSTANCES, METHODS, SEED, EPS_G, MAX_ITER

# ── output dirs ───────────────────────────────────────────────────────────────
_RES    = _ROOT / "results" / "5_large_scale_synthesis"
RAW_DIR = _RES
SUM_DIR = _RES
LAT_DIR = _RES
FIG_DIR = _RES / "figures"
MD_DIR  = _RES
_RES.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SynthResult:
    family:            str
    label:             str
    problem:           str
    n:                 int
    method:            str
    x0_label:          str
    success:           bool
    termination_reason: str
    iterations:        int
    rejected_iter:     int
    runtime_sec:       float
    final_f:           float
    final_grad_norm:   float
    sigma0:            float
    sigma_final:       float
    numerical_failure: bool

    def to_row(self) -> dict:
        return {k: (round(v, 8) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "family", "label", "problem", "n", "method", "x0_label",
            "success", "termination_reason",
            "iterations", "rejected_iter", "runtime_sec",
            "final_f", "final_grad_norm",
            "sigma0", "sigma_final", "numerical_failure",
        ]


# ── single run ────────────────────────────────────────────────────────────────

RUN_TIMEOUT = 120  # seconds per run; exceeded → record as timeout


def run_one(inst, method: str, x0_label: str, x0: np.ndarray) -> SynthResult:
    import signal

    def _alarm(signum, frame):
        raise TimeoutError(f"run exceeded {RUN_TIMEOUT}s")

    f_raw, grad_raw, hess_raw, meta = get_problem(
        inst.problem, inst.n, seed=SEED, problem_kwargs=inst.problem_kwargs
    )
    counter = EvalCounter(f_raw, grad_raw, hess_raw)
    L3 = meta.get("L3")

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(RUN_TIMEOUT)
    t0 = time.perf_counter()
    try:
        raw = run_and_extract(
            method=method,
            counter=counter,
            x0=x0.copy(),
            eps_g=EPS_G,
            max_iter=MAX_ITER,
            L3=L3,
            solver_kwargs={},
        )
        signal.alarm(0)
    except Exception as exc:
        signal.alarm(0)
        elapsed = time.perf_counter() - t0
        reason = "timeout" if isinstance(exc, TimeoutError) else f"error:{str(exc)[:80]}"
        return SynthResult(
            family=inst.family, label=inst.label, problem=inst.problem, n=inst.n,
            method=method, x0_label=x0_label,
            success=False, termination_reason=reason,
            iterations=0, rejected_iter=0, runtime_sec=elapsed,
            final_f=float("nan"), final_grad_norm=float("nan"),
            sigma0=float("nan"), sigma_final=float("nan"),
            numerical_failure=True,
        )

    elapsed = time.perf_counter() - t0
    x_final   = raw["x_final"]
    num_fail  = not np.all(np.isfinite(x_final))
    final_f   = float(f_raw(x_final))   if not num_fail else float("nan")
    final_gn  = float(norm(grad_raw(x_final))) if not num_fail else float("nan")
    success   = np.isfinite(final_gn) and final_gn <= EPS_G

    return SynthResult(
        family=inst.family, label=inst.label, problem=inst.problem, n=inst.n,
        method=method, x0_label=x0_label,
        success=success,
        termination_reason=raw["stop_reason"],
        iterations=raw["outer_iterations"],
        rejected_iter=raw["rejected_iterations"],
        runtime_sec=elapsed,
        final_f=final_f,
        final_grad_norm=final_gn,
        sigma0=float(raw["sigma0"]) if np.isfinite(float(raw["sigma0"])) else float("nan"),
        sigma_final=float(raw["sigma_final"]) if np.isfinite(float(raw["sigma_final"])) else float("nan"),
        numerical_failure=num_fail,
    )


# ── full experiment ────────────────────────────────────────────────────────────

def run_all() -> List[SynthResult]:
    results = []
    for inst in ALL_INSTANCES:
        print(f"\n  [{inst.family}] {inst.label}")
        for method in METHODS:
            for x0_label, x0 in inst.x0s.items():
                r = run_one(inst, method, x0_label, x0)
                results.append(r)
            ok  = sum(1 for r in results
                      if r.label == inst.label and r.method == method and r.success)
            tot = len(inst.x0s)
            print(f"    {method:8s}: {ok}/{tot} converged")
    return results


# ── CSV output ────────────────────────────────────────────────────────────────

def write_raw_csv(results: List[SynthResult]) -> None:
    path = RAW_DIR / "raw_results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SynthResult.csv_fieldnames())
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())
    print(f"\nRaw CSV    → {path}  ({len(results)} rows)")


def _med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def write_aggregated_csv(results: List[SynthResult]) -> None:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.label, r.n, r.method)].append(r)

    fieldnames = [
        "family", "label", "n", "method",
        "n_runs", "success_rate",
        "iters_median", "rejected_median", "runtime_median",
        "final_f_median", "final_grad_norm_median",
    ]
    rows = []
    for (fam, lbl, n, meth), grp in sorted(groups.items()):
        ok    = [r for r in grp if r.success]
        rows.append({
            "family":                fam,
            "label":                 lbl,
            "n":                     n,
            "method":                meth,
            "n_runs":                len(grp),
            "success_rate":          round(len(ok) / len(grp), 3),
            "iters_median":          _med([r.iterations for r in grp if r.iterations > 0]),
            "rejected_median":       _med([r.rejected_iter for r in grp]),
            "runtime_median":        _med([r.runtime_sec for r in grp]),
            "final_f_median":        _med([r.final_f for r in ok]),
            "final_grad_norm_median": _med([r.final_grad_norm for r in ok]),
        })

    path = SUM_DIR / "aggregated.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Aggregated → {path}  ({len(rows)} rows)")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def write_latex_tables(results: List[SynthResult]) -> None:
    from make_latex_tables import write_all_tables
    write_all_tables(results, LAT_DIR)


# ── scaling figures ───────────────────────────────────────────────────────────

def write_figures(results: List[SynthResult]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.n, r.method)].append(r)

    markers = {"Newton": "o", "NCR": "s", "ARC": "^", "ACRN": "D"}
    colors  = {"Newton": "C0", "NCR": "C1", "ARC": "C2", "ACRN": "C3"}

    for family in ["logsumexp", "rosenbrock", "ill_conditioned"]:
        fam_results = [r for r in results if r.family == family]
        if not fam_results:
            continue
        dims = sorted(set(r.n for r in fam_results))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in METHODS:
            iters = []
            times = []
            ns    = []
            for n in dims:
                grp = groups.get((family, n, method), [])
                ok  = [r for r in grp if r.success and r.iterations > 0]
                if ok:
                    ns.append(n)
                    iters.append(_med([r.iterations for r in ok]))
                    times.append(_med([r.runtime_sec for r in ok]))
            if ns:
                axes[0].plot(ns, iters, marker=markers[method],
                             color=colors[method], label=method, linewidth=1.5)
                axes[1].plot(ns, times, marker=markers[method],
                             color=colors[method], label=method, linewidth=1.5)

        for ax, ylabel in zip(axes, ["Median iterations", "Median runtime (s)"]):
            ax.set_xlabel("Dimension $n$")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(family.replace("_", " ").title(), fontsize=11)
        fig.tight_layout()
        out = FIG_DIR / f"scaling_{family}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure → {out}")


# ── markdown ──────────────────────────────────────────────────────────────────

def write_markdown(results: List[SynthResult]) -> None:
    from datetime import datetime
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.label, r.method)].append(r)

    lines = [
        "# Large-Scale Synthesis Experiment",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Experimental Defaults",
        "",
        "| Setting | Value | Source |",
        "|---------|-------|--------|",
        f"| Gradient tolerance (ε_g) | 1e-7 | experiments/2 and 3 |",
        f"| Max iterations | {MAX_ITER} | all earlier experiments |",
        f"| Seed | {SEED} | all earlier experiments |",
        f"| σ₀ policy | max(2·L3, 0.5) | adapters.py default |",
        f"| Methods | {METHODS} | |",
        "",
        "## Benchmark Families",
        "",
        "| Family | Purpose | Dimensions |",
        "|--------|---------|------------|",
        "| LogSumExp | Convex scalability; fair CRN vs ACRN | 10, 20, 50, 100 |",
        "| Rosenbrock | Nonconvex scalability; narrow valley | 2, 10, 20, 50, 100 |",
        "| Quadratic (ill-cond.) | Convex, ill-conditioned; reconnect to Newton fragility | n∈{10,50,100}, κ∈{1e2,1e4,1e6} |",
        "",
        "## Starting Points",
        "",
        "Two starts per instance: `standard` and `benign`.",
        "Rosenbrock: standard = (−1,…,−1), benign = (0.5,…,0.5).",
        "Others: standard = N(0,1) (seed 1042), benign = 0.",
        "",
        "## Results Summary",
        "",
    ]

    for family in ["logsumexp", "rosenbrock", "ill_conditioned"]:
        fam_results = [r for r in results if r.family == family]
        if not fam_results:
            continue
        lines.append(f"### {family.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Problem | Method | Rate | Med.k | Med.t(s) |")
        lines.append("|---------|--------|------|-------|----------|")
        labels = sorted(set(r.label for r in fam_results),
                        key=lambda l: next(r.n for r in fam_results if r.label == l))
        for lbl in labels:
            for meth in METHODS:
                grp = groups.get((family, lbl, meth), [])
                if not grp:
                    continue
                ok   = [r for r in grp if r.success]
                rate = len(ok) / len(grp)
                med_k = _med([r.iterations for r in grp if r.iterations > 0])
                med_t = _med([r.runtime_sec for r in grp])
                k_str = f"{med_k:.0f}" if np.isfinite(med_k) else "---"
                t_str = f"{med_t:.3f}" if np.isfinite(med_t) else "---"
                lines.append(f"| {lbl} | {meth} | {rate:.2f} | {k_str} | {t_str} |")
        lines.append("")

    lines += [
        "## Output Files",
        "",
        "| Path | Description |",
        "|------|-------------|",
        "| `raw/raw_results.csv` | One row per (instance, method, start) |",
        "| `summary/aggregated.csv` | Grouped medians per (instance, method) |",
        "| `latex/table_{family}.tex` | Compact LaTeX tables per family |",
        "| `latex/table_acrn_gain.tex` | ACRN vs CRN iteration/runtime gain (convex) |",
        "| `figures/scaling_{family}.pdf` | Iteration and runtime vs dimension |",
        "",
    ]

    path = MD_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown   → {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Large-scale synthesis experiment")
    print(f"Methods  : {METHODS}")
    print(f"eps_g    : {EPS_G}   max_iter : {MAX_ITER}   seed : {SEED}")
    print(f"Instances: {len(ALL_INSTANCES)}")
    total = sum(len(inst.x0s) for inst in ALL_INSTANCES) * len(METHODS)
    print(f"Total runs: {total}")
    print("=" * 60)

    results = run_all()

    write_raw_csv(results)
    write_aggregated_csv(results)
    write_latex_tables(results)
    write_figures(results)
    write_markdown(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
