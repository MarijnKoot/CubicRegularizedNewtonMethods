"""
make_latex_tables.py — Compact LaTeX tables for the synthesis experiment.

Tables produced:
  table_logsumexp.tex      — convex scalability
  table_rosenbrock.tex     — nonconvex scalability
  table_ill_conditioned.tex — ill-conditioned quadratics
  table_acrn_gain.tex      — ACRN vs CRN gain on LogSumExp
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
for _p in [_ROOT, _ROOT / "src"]:
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from benchmarks import METHODS, EPS_G, MAX_ITER


def _med(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def _fmt_rate(v):
    return f"{v:.2f}"

def _fmt_k(v):
    return f"{v:.0f}" if np.isfinite(v) else "---"

def _fmt_rej(v):
    return f"{v:.0f}" if np.isfinite(v) else "---"

def _fmt_t(v):
    if not np.isfinite(v):
        return "---"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"

def _fmt_gn(v):
    return f"{v:.1e}" if np.isfinite(v) else "---"


def _main_table(
    results,
    family: str,
    caption: str,
    label: str,
    show_methods: List[str],
    out_path: Path,
) -> None:
    """
    Compact table: rows = (problem, method), cols = Rate | k | Rej | Time | ‖g‖
    """
    groups = defaultdict(list)
    for r in results:
        if r.family == family:
            groups[(r.label, r.n, r.method)].append(r)

    if not groups:
        return

    labels_ns = sorted(
        set((r.label, r.n) for r in results if r.family == family),
        key=lambda ln: (ln[1], ln[0]),
    )

    col_spec = "|l|l|r|r|r|r|r|"
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline")
    lines.append(
        r"    \textbf{Problem} & \textbf{Method} & \textbf{Rate} "
        r"& \textbf{$k$} & \textbf{Rej.} & \textbf{Time\,(s)} "
        r"& \textbf{$\|\nabla f\|$} \\"
    )
    lines.append(r"    \hline\hline")

    prev_lbl = None
    for lbl, n in labels_ns:
        first_in_block = True
        for meth in show_methods:
            grp = groups.get((lbl, n, meth), [])
            if not grp:
                continue
            ok   = [r for r in grp if r.success]
            rate = len(ok) / len(grp)
            med_k   = _med([r.iterations for r in grp if r.iterations > 0])
            med_rej = _med([r.rejected_iter for r in grp])
            med_t   = _med([r.runtime_sec for r in grp])
            med_gn  = _med([r.final_grad_norm for r in ok])

            row_lbl = lbl if first_in_block else ""
            lines.append(
                f"    {row_lbl} & {meth} & {_fmt_rate(rate)} "
                f"& {_fmt_k(med_k)} & {_fmt_rej(med_rej)} "
                f"& {_fmt_t(med_t)} & {_fmt_gn(med_gn)} \\\\"
            )
            first_in_block = False

        lines.append(r"    \hline")

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX → {out_path}")


def _acrn_gain_table(results, out_path: Path) -> None:
    """
    Small table: dimension | CRN k | ACRN k | Δk | CRN t | ACRN t | Δt
    Only for LogSumExp (convex family) where ACRN is meaningful.
    """
    groups = defaultdict(list)
    for r in results:
        if r.family == "logsumexp":
            groups[(r.n, r.method)].append(r)

    dims = sorted(set(n for n, _ in groups))
    if not dims:
        return

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{ACRN vs.\ NCR iteration and runtime gain on LogSumExp "
        r"(median over starts; ``---'' = no convergence).}"
    )
    lines.append(r"  \label{tab:acrn_gain}")
    lines.append(r"  \begin{tabular}{|r|r|r|r|r|r|r|}")
    lines.append(r"    \hline")
    lines.append(
        r"    $n$ & $k_\mathrm{NCR}$ & $k_\mathrm{ACRN}$ & $\Delta k$ "
        r"& $t_\mathrm{NCR}$\,(s) & $t_\mathrm{ACRN}$\,(s) & $\Delta t$ \\"
    )
    lines.append(r"    \hline\hline")

    for n in dims:
        ncr_grp  = groups.get((n, "NCR"),  [])
        acrn_grp = groups.get((n, "ACRN"), [])
        ncr_ok   = [r for r in ncr_grp  if r.success]
        acrn_ok  = [r for r in acrn_grp if r.success]

        k_ncr  = _med([r.iterations for r in ncr_ok])
        k_acrn = _med([r.iterations for r in acrn_ok])
        t_ncr  = _med([r.runtime_sec for r in ncr_grp])
        t_acrn = _med([r.runtime_sec for r in acrn_grp])

        dk = f"{k_acrn/k_ncr:.2f}×" if (np.isfinite(k_ncr) and np.isfinite(k_acrn) and k_ncr > 0) else "---"
        dt = f"{t_acrn/t_ncr:.2f}×" if (np.isfinite(t_ncr) and np.isfinite(t_acrn) and t_ncr > 0) else "---"

        lines.append(
            f"    {n} & {_fmt_k(k_ncr)} & {_fmt_k(k_acrn)} & {dk} "
            f"& {_fmt_t(t_ncr)} & {_fmt_t(t_acrn)} & {dt} \\\\"
        )
        lines.append(r"    \hline")

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  LaTeX → {out_path}")


def write_all_tables(results, lat_dir: Path) -> None:
    # Convex: all methods
    _main_table(
        results,
        family="logsumexp",
        caption=(
            "Synthesis results on LogSumExp (convex, scalable). "
            f"$\\varepsilon_g = 10^{{-7}}$, max\\_iter = {MAX_ITER}. "
            "Median over 2 starting points."
        ),
        label="tab:synth_logsumexp",
        show_methods=METHODS,
        out_path=lat_dir / "table_logsumexp.tex",
    )

    # Nonconvex: Newton / NCR / ARC (ACRN secondary)
    _main_table(
        results,
        family="rosenbrock",
        caption=(
            "Synthesis results on Rosenbrock (nonconvex, chain). "
            f"$\\varepsilon_g = 10^{{-7}}$, max\\_iter = {MAX_ITER}. "
            "Median over 2 starting points."
        ),
        label="tab:synth_rosenbrock",
        show_methods=METHODS,
        out_path=lat_dir / "table_rosenbrock.tex",
    )

    # Ill-conditioned
    _main_table(
        results,
        family="ill_conditioned",
        caption=(
            "Synthesis results on ill-conditioned quadratics. "
            f"$\\varepsilon_g = 10^{{-7}}$, max\\_iter = {MAX_ITER}. "
            "Median over 2 starting points."
        ),
        label="tab:synth_illcond",
        show_methods=METHODS,
        out_path=lat_dir / "table_ill_conditioned.tex",
    )

    # ACRN gain table
    _acrn_gain_table(results, lat_dir / "table_acrn_gain.tex")
