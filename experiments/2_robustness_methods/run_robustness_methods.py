"""
run_robustness_methods.py — Robustness of optimization methods w.r.t. problem geometry.

Usage (from repo root, with venv active):
    python experiments/2_robustness_methods/run_robustness_methods.py

Outputs under experiments/2_robustness_methods/results/
    raw/raw_results.csv
    summary/aggregated.csv
    summary/latex_tables.txt
    figures/...
    markdown/summary.md
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from benchmarks        import ALL_BENCHMARKS, METHODS, SEED, EPS_G, MAX_ITER

# ── output dirs ───────────────────────────────────────────────────────────────
_RES    = _ROOT / "results" / "2_robustness_methods"
RAW_DIR = _RES
SUM_DIR = _RES
FIG_DIR = _RES / "figures"
MD_DIR  = _RES
_RES.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RobustnessResult:
    family:       str
    label:        str
    problem:      str
    n:            int
    method:       str
    x0_label:     str
    success:      bool
    termination_reason: str
    iterations:   int
    runtime_sec:  float
    final_f:      float
    final_grad_norm: float
    rejected_iter: int
    numerical_failure: bool   # NaN/Inf detected in iterates

    def to_row(self) -> dict:
        return {
            "family":             self.family,
            "label":              self.label,
            "problem":            self.problem,
            "n":                  self.n,
            "method":             self.method,
            "x0_label":           self.x0_label,
            "success":            int(self.success),
            "termination_reason": self.termination_reason,
            "iterations":         self.iterations,
            "runtime_sec":        round(self.runtime_sec, 6),
            "final_f":            self.final_f,
            "final_grad_norm":    self.final_grad_norm,
            "rejected_iter":      self.rejected_iter,
            "numerical_failure":  int(self.numerical_failure),
        }

    @staticmethod
    def csv_fieldnames() -> List[str]:
        return [
            "family", "label", "problem", "n", "method", "x0_label",
            "success", "termination_reason",
            "iterations", "runtime_sec",
            "final_f", "final_grad_norm",
            "rejected_iter", "numerical_failure",
        ]


# ── single run ────────────────────────────────────────────────────────────────

def run_one(
    family: str,
    label: str,
    problem_name: str,
    n: int,
    problem_kwargs: dict,
    method: str,
    x0_label: str,
    x0: np.ndarray,
) -> RobustnessResult:

    f_raw, grad_raw, hess_raw, meta = get_problem(
        problem_name, n, seed=SEED, problem_kwargs=problem_kwargs
    )

    if method == "ACRN" and not meta.get("convex", False):
        return RobustnessResult(
            family=family, label=label, problem=problem_name, n=n,
            method=method, x0_label=x0_label,
            success=False, termination_reason="skip:nonconvex",
            iterations=0, runtime_sec=float("nan"),
            final_f=float("nan"), final_grad_norm=float("nan"),
            rejected_iter=0, numerical_failure=False,
        )

    counter = EvalCounter(f_raw, grad_raw, hess_raw)
    L3 = meta.get("L3")

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
    except Exception as exc:
        return RobustnessResult(
            family=family, label=label, problem=problem_name, n=n,
            method=method, x0_label=x0_label,
            success=False, termination_reason=f"error:{str(exc)[:80]}",
            iterations=0, runtime_sec=float("nan"),
            final_f=float("nan"), final_grad_norm=float("nan"),
            rejected_iter=0, numerical_failure=True,
        )

    x_final    = raw["x_final"]
    num_fail   = not np.all(np.isfinite(x_final))
    final_f    = float(f_raw(x_final)) if not num_fail else float("nan")
    final_gnorm = float(norm(grad_raw(x_final))) if not num_fail else float("nan")
    success    = np.isfinite(final_gnorm) and final_gnorm <= EPS_G

    return RobustnessResult(
        family=family, label=label, problem=problem_name, n=n,
        method=method, x0_label=x0_label,
        success=success,
        termination_reason=raw["stop_reason"],
        iterations=raw["outer_iterations"],
        runtime_sec=raw["elapsed"],
        final_f=final_f,
        final_grad_norm=final_gnorm,
        rejected_iter=raw["rejected_iterations"],
        numerical_failure=num_fail,
    )


# ── full experiment ────────────────────────────────────────────────────────────

def run_all() -> List[RobustnessResult]:
    results = []
    for inst in ALL_BENCHMARKS:
        print(f"\n  [{inst.family}] {inst.label}  (n={inst.n})")
        for method in METHODS:
            for x0_label, x0 in inst.x0s.items():
                r = run_one(
                    family=inst.family,
                    label=inst.label,
                    problem_name=inst.problem,
                    n=inst.n,
                    problem_kwargs=inst.problem_kwargs,
                    method=method,
                    x0_label=x0_label,
                    x0=x0,
                )
                results.append(r)
            ok = sum(1 for r in results
                     if r.label == inst.label and r.method == method and r.success)
            tot = len(inst.x0s)
            print(f"    {method:8s}: {ok}/{tot} converged")
    return results


# ── CSV output ────────────────────────────────────────────────────────────────

def write_raw_csv(results: List[RobustnessResult]) -> Path:
    path = RAW_DIR / "raw_results.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RobustnessResult.csv_fieldnames())
        w.writeheader()
        for r in results:
            w.writerow(r.to_row())
    print(f"\nRaw CSV → {path}  ({len(results)} rows)")
    return path


def write_aggregated_csv(results: List[RobustnessResult]) -> Path:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.label, r.method)].append(r)

    fieldnames = [
        "family", "label", "method",
        "n_runs", "success_rate",
        "iters_median", "iters_max",
        "runtime_median",
        "final_f_median", "final_grad_norm_median",
        "rejected_median", "failure_count",
        "sensitivity_ratio",
    ]
    rows = []
    for (fam, lbl, meth), grp in sorted(groups.items()):
        ok    = [r for r in grp if r.success]
        iters = [r.iterations for r in grp if r.iterations > 0]
        iters_ok = [r.iterations for r in ok]
        rts   = [r.runtime_sec for r in grp if np.isfinite(r.runtime_sec)]
        fvals = [r.final_f for r in ok if np.isfinite(r.final_f)]
        gnorms= [r.final_grad_norm for r in ok if np.isfinite(r.final_grad_norm)]
        rejs  = [r.rejected_iter for r in grp]
        fails = sum(1 for r in grp if r.numerical_failure or not r.success)
        sens  = (max(iters_ok)/min(iters_ok)
                 if len(iters_ok) >= 2 and min(iters_ok) > 0 else float("nan"))
        rows.append({
            "family":               fam,
            "label":                lbl,
            "method":               meth,
            "n_runs":               len(grp),
            "success_rate":         round(len(ok)/len(grp), 3),
            "iters_median":         float(np.median(iters)) if iters else float("nan"),
            "iters_max":            max(iters) if iters else 0,
            "runtime_median":       float(np.median(rts)) if rts else float("nan"),
            "final_f_median":       float(np.median(fvals)) if fvals else float("nan"),
            "final_grad_norm_median": float(np.median(gnorms)) if gnorms else float("nan"),
            "rejected_median":      float(np.median(rejs)) if rejs else 0,
            "failure_count":        fails,
            "sensitivity_ratio":    round(sens, 2) if np.isfinite(sens) else float("nan"),
        })

    path = SUM_DIR / "aggregated.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Aggregated → {path}  ({len(rows)} rows)")
    return path


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def write_latex_tables(results: List[RobustnessResult]) -> Path:
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r.family, r.label, r.method)].append(r)

    def _rate(grp):
        if not grp or all(r.termination_reason == "skip:nonconvex" for r in grp):
            return None
        ok = [r for r in grp if r.success]
        return len(ok) / len(grp)

    def _med_iters(grp):
        ok = [r for r in grp if r.success and r.iterations > 0]
        return np.median([r.iterations for r in ok]) if ok else float("nan")

    def _med_gnorm(grp):
        ok = [r for r in grp if r.success and np.isfinite(r.final_grad_norm)]
        return np.median([r.final_grad_norm for r in ok]) if ok else float("nan")

    families = {
        "ill_conditioned": "Ill-conditioned quadratics",
        "rosenbrock":      "Rosenbrock (narrow valley)",
        "dixon_price":     "Dixon-Price (coupled variables)",
        "rastrigin":       "Rastrigin (multimodal)",
    }

    n_runs = max((len(v) for v in groups.values()), default=0)

    lines = []
    for fam_key, fam_title in families.items():
        fam_results = [r for r in results if r.family == fam_key]
        if not fam_results:
            continue

        labels_in_fam = sorted(
            set(r.label for r in fam_results),
            key=lambda l: next((r.n for r in fam_results if r.label == l), 0),
        )
        methods_here = [m for m in METHODS
                        if any(r.method == m and r.family == fam_key
                               and r.termination_reason != "skip:nonconvex"
                               for r in fam_results)]

        # One column per method: "rate / med_k / med_f(x*)"
        # Grid style: vertical bars + \hline rows
        n_m = len(methods_here)
        col_spec = "|l|" + "|".join("c" for _ in methods_here) + "|"

        lines.append(f"\n% ── {fam_title} ──────────────────────────────")
        lines.append(r"\begin{table}[ht]")
        lines.append(r"  \centering")
        caption = (
            f"Robustness of methods on \\textit{{{fam_title}}} "
            f"({n_runs} starting points per instance, $\\varepsilon_g = 10^{{-7}}$). "
            r"Each cell: \textit{rate} / \textit{med.\ $k$} / \textit{med.\ $f(x^*)$}. "
            r"Rate 1.00 = all starts converged; `\texttt{---}' = not applicable."
        )
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{tab:robust_{fam_key}}}")
        lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"    \hline")
        lines.append("    \\textbf{Problem} & " + " & ".join(f"\\textbf{{{m}}}" for m in methods_here) + r" \\")
        lines.append(r"    \hline\hline")

        def _med_f(grp):
            ok = [r for r in grp if r.success and np.isfinite(r.final_f)]
            return np.median([r.final_f for r in ok]) if ok else float("nan")

        for lbl in labels_in_fam:
            cells = []
            for meth in methods_here:
                grp = groups.get((fam_key, lbl, meth), [])
                rate = _rate(grp)
                if rate is None:
                    cells.append("---")
                else:
                    med_k = _med_iters(grp)
                    med_f = _med_f(grp)
                    k_str = f"{med_k:.0f}" if np.isfinite(med_k) else "---"
                    f_str = f"{med_f:.2e}" if np.isfinite(med_f) else "---"
                    cells.append(f"{rate:.2f} / {k_str} / {f_str}")
            lines.append(f"    {lbl} & " + " & ".join(cells) + r" \\")
            lines.append(r"    \hline")

        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

    path = SUM_DIR / "latex_tables.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LaTeX     → {path}")
    return path


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Robustness experiment — problem geometry")
    print(f"Methods  : {METHODS}")
    print(f"eps_g    : {EPS_G}   max_iter : {MAX_ITER}")
    print(f"Seed     : {SEED}")
    print(f"Instances: {len(ALL_BENCHMARKS)} benchmark instances")
    print("=" * 60)

    results = run_all()

    write_raw_csv(results)
    write_aggregated_csv(results)
    write_latex_tables(results)

    # ── plots ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    from plot_robustness_methods import plot_all
    plot_all(results)

    # ── markdown ─────────────────────────────────────────────────────────────
    from plot_robustness_methods import write_markdown
    write_markdown(results)

    # ── print LaTeX to stdout ─────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("LaTeX tables  (copy into thesis)")
    print("=" * 60)
    print((SUM_DIR / "latex_tables.txt").read_text(encoding="utf-8"))

    print("\nDone.")


if __name__ == "__main__":
    main()
