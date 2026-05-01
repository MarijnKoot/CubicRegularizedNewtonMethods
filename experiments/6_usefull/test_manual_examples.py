"""
Manual example traces for CRN, ARC, and ACRN.

Runs each solver for a fixed number of iterations on a 1-D convex test
function and prints the per-iteration table used to verify thesis hand
computations.  Also saves a combined path plot to results/6_usefull/figures/.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = SCRIPT_DIR.parent.parent  # experiments/6_usefull -> experiments -> repo root
SRC = str(ROOT / "src")
for p in (SRC, str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from methods.NCR  import CubicNewton, CRNOptions
from methods.ARC  import AdaptiveCubicNewton, ARCParams
from methods.ACRN import AcceleratedCubicNewton

ITERATIONS = 4
FIG_DIR = ROOT / "results" / "6_usefull" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=6, suppress=True)


# ── test function ─────────────────────────────────────────────────────────────

def make_convex(mu=1.0):
    """f(x) = 0.5 μ x² + 1 - cos(x),  x* = 0,  f* = 0."""
    def f(x):
        x = float(np.asarray(x).ravel()[0])
        return 0.5 * mu * x * x + 1.0 - np.cos(x)
    def grad(x):
        x = float(np.asarray(x).ravel()[0])
        return np.array([mu * x + np.sin(x)])
    def hess(x):
        x = float(np.asarray(x).ravel()[0])
        return np.array([[mu + np.cos(x)]])
    return f, grad, hess


def make_oscillating():
    """f(x) = 10 + x² - cos(2πx),  x* = 0,  f* = 9.
    Non-convex: f''(x) = 2 + 4π²cos(2πx) can be negative.
    ACRN is not applicable; only NCR and ARC are run.
    """
    def f(x):
        x = float(np.asarray(x).ravel()[0])
        return 10.0 + x * x - np.cos(2.0 * np.pi * x)
    def grad(x):
        x = float(np.asarray(x).ravel()[0])
        return np.array([2.0 * x + 2.0 * np.pi * np.sin(2.0 * np.pi * x)])
    def hess(x):
        x = float(np.asarray(x).ravel()[0])
        return np.array([[2.0 + 4.0 * np.pi ** 2 * np.cos(2.0 * np.pi * x)]])
    return f, grad, hess


# ── table printing ────────────────────────────────────────────────────────────

def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, np.ndarray):
        v = float(v.ravel()[0])
    if isinstance(v, (float, int, np.floating)):
        return f"{float(v): .6f}"
    return str(v)

def print_table(title, columns, rows):
    print(f"\n{'─'*60}\n{title}\n{'─'*60}")
    if not rows:
        print(" (empty)")
        return
    widths = [max(len(c), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  ".join("─" * w for w in widths))
    for r in rows:
        print("  ".join(_fmt(r.get(c, "")).rjust(w) for c, w in zip(columns, widths)))


_LATEX_COL = {
    "k":          "$k$",
    "x_k":        "$x_k$",
    "σ_k":        r"$\sigma_k$",
    "g_k":        "$g_k$",
    "H_k":        "$H_k$",
    "h_k":        "$h_k$",
    "f(x_k)":     "$f(x_k)$",
    "x_{k+1}":    "$x_{k+1}$",
    "f(x_k+h_k)": "$f(x_k+h_k)$",
    "Δm_k":       r"$\Delta m_k$",
    "ρ_k":        r"$\rho_k$",
    "accept":     "accept",
    "y_k":        "$y_k$",
    "f'(x_k)":    "$f'(x_k)$",
    "f''(x_k)":   "$f''(x_k)$",
}


def _fmt_latex(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{---}"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, np.ndarray):
        v = float(v.ravel()[0])
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6f}"
    return str(v)


def print_latex_table(columns, rows):
    col_fmt = "|r|" + "r" * (len(columns) - 1) + "|"
    headers = [_LATEX_COL.get(c, c) for c in columns]
    print(r"\begin{tabular}{" + col_fmt + "}")
    print(r"\hline")
    print(" & ".join(headers) + r" \\")
    print(r"\hline")
    for r in rows:
        print(" & ".join(_fmt_latex(r.get(c, float("nan"))) for c in columns) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")


# ── solvers ───────────────────────────────────────────────────────────────────

def run_crn(f, grad, hess, x0, sigma0):
    solver = CubicNewton(f, grad, hess, options=CRNOptions(
        sigma0=sigma0, sigma_min=sigma0, sigma_max=sigma0,
        tol_grad=0.0, tol_step=0.0, max_iter=ITERATIONS, verbose=False,
    ))
    x_final = solver.run(x0.copy())

    # NCR log stores x=x_{k+1} (post-step) and sigma=sigma_{k+1} (post-update).
    # Reconstruct pre-step x_k and pre-update sigma_k from the running state.
    rows = []
    x_prev  = x0.copy()
    sig_prev = sigma0
    for e in solver.log:
        hk      = e.get("h")
        x_next  = e["x"]          # = x_{k+1}
        rows.append({
            "k":        e["iter"],
            "x_k":      float(x_prev.ravel()[0]),
            "σ_k":      sig_prev,
            "g_k":      float(np.asarray(e.get("g", grad(x_prev))).ravel()[0]),
            "H_k":      float(np.asarray(e.get("H", hess(x_prev))).ravel()[0]),
            "h_k":      float(hk.ravel()[0]) if hk is not None else float("nan"),
            "f(x_k)":   float(e["f"]),
            "x_{k+1}":  float(x_next.ravel()[0]),
            "accept":   e.get("accepted", True),
        })
        x_prev  = x_next
        sig_prev = e["sigma"]     # = sigma_{k+1}, becomes sigma_k next round

    iterates = [float(x0.ravel()[0])] + [r["x_{k+1}"] for r in rows]
    return rows, iterates, float(x_final.ravel()[0])


def run_arc(f, grad, hess, x0, sigma0):
    solver = AdaptiveCubicNewton(f, grad, hess, params=ARCParams(
        sigma0=sigma0, tol_grad=0.0, tol_step=0.0,
        max_iter=ITERATIONS, verbose=False,
    ), step_method="secular")
    x_final = solver.run(x0.copy())

    rows = []
    for e in solver.log:
        xk = e["x"]; hk = e.get("h"); xt = e.get("x_trial", xk + hk if hk is not None else xk)
        rows.append({
            "k":           e["iter"],
            "x_k":         float(xk.ravel()[0]),
            "σ_k":         e["sigma"],
            "h_k":         float(hk.ravel()[0]) if hk is not None else float("nan"),
            "f(x_k)":      f(xk),
            "f(x_k+h_k)":  f(xt),
            "Δm_k":        e.get("predicted_reduction", float("nan")),
            "ρ_k":         e.get("rho", float("nan")),
            "accept":      e.get("accepted", True),
        })

    iterates = [float(x0.ravel()[0])]
    for e in solver.log:
        if e.get("accepted", True):
            xk = e["x"]; hk = e.get("h")
            if hk is not None:
                iterates.append(float((xk + hk).ravel()[0]))
    return rows, iterates, float(x_final.ravel()[0])


def run_acrn(f, grad, hess, x0, L3):
    solver = AcceleratedCubicNewton(f, grad, hess,
        L3=L3, sigma=L3, tol_grad=0.0, max_iter=ITERATIONS, verbose=False)
    result = solver.run(x0.copy())
    x_final, info = result if isinstance(result, tuple) else (result, {})
    hist = info.get("history", [])

    rows = []
    x_prev = x0
    for row in hist:
        yk = row.yk; hk = row.hk; xnew = yk + hk
        rows.append({
            "k":         row.k,
            "x_k":       float(x_prev.ravel()[0]),
            "f(x_k)":    f(x_prev),
            "f'(x_k)":   float(np.asarray(grad(x_prev)).ravel()[0]),
            "f''(x_k)":  float(np.asarray(hess(x_prev)).ravel()[0]),
            "y_k":       float(yk.ravel()[0]),
            "h_k":       float(hk.ravel()[0]),
            "x_{k+1}":   float(xnew.ravel()[0]),
        })
        x_prev = xnew

    iterates = [float(x0.ravel()[0])] + [r["x_{k+1}"] for r in rows]
    return rows, iterates, float(np.asarray(x_final).ravel()[0])


# ── per-method plots ──────────────────────────────────────────────────────────

METHOD_STYLE = {
    "NCR":  dict(color="C0", marker="o", ls="-",  label="NCR"),
    "ARC":  dict(color="C1", marker="s", ls="--", label="ARC"),
    "ACRN": dict(color="C2", marker="^", ls="-.", label="ACRN"),
}


def plot_path_single(f, name, iterates, x_range=None):
    """f(x) curve + numbered iterate dots with arrows for one method."""
    st = METHOD_STYLE[name]
    xs = list(iterates)
    if x_range is not None:
        xgrid = np.linspace(x_range[0], x_range[1], 600)
    else:
        lo, hi = min(xs), max(xs)
        span = max(hi - lo, 0.5)
        xgrid = np.linspace(lo - 0.4 * span, hi + 0.4 * span, 600)
    fgrid = [f(x) for x in xgrid]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xgrid, fgrid, color="0.3", lw=1.5, label="$f(x)$", zorder=1)
    ys = [f(x) for x in xs]
    ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=1.6, zorder=2)
    for i in range(len(xs) - 1):
        ax.annotate("",
            xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color=st["color"],
                            lw=1.2, mutation_scale=10), zorder=3)
    for k, (x, y) in enumerate(zip(xs, ys)):
        ax.scatter(x, y, color=st["color"], s=40, zorder=4)
    if xs:
        ax.annotate("$x_0$", (xs[0], ys[0]),
            textcoords="offset points", xytext=(-4, 8),
            fontsize=9, color=st["color"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))
        ax.annotate(f"$x_{len(xs)-1}$", (xs[-1], ys[-1]),
            textcoords="offset points", xytext=(-4, 8),
            fontsize=9, color=st["color"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))
    ax.axvline(0, color="0.6", ls=":", lw=1, label="$x^*=0$")
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$f(x)$", fontsize=11)
    ax.set_title(f"{st['label']} — optimization path", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"manual_example_{name.lower()}_path.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def plot_path_oscillating(f, name, main_iterates, extra_iterates, x_range=(-1, 3)):
    """f(x) curve + main path in method colour + extra local-convergence paths in dark blue."""
    st = METHOD_STYLE[name]
    xgrid = np.linspace(x_range[0], x_range[1], 800)
    fgrid = [f(x) for x in xgrid]
    dark_blues = ["#c0392b", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xgrid, fgrid, color="0.3", lw=1.5, label="$f(x)$", zorder=1)

    for (its, x0_label), color in zip(extra_iterates, dark_blues):
        xs = list(its)
        ys = [f(x) for x in xs]
        ax.plot(xs, ys, color=color, ls="--", lw=1.3, zorder=2)
        for i in range(len(xs) - 1):
            ax.annotate("",
                xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.0, mutation_scale=8), zorder=3)
        ax.scatter(xs, ys, color=color, s=30, zorder=4)
        if xs:
            ax.annotate(f"$x_0={x0_label}$", (xs[0], ys[0]),
                textcoords="offset points", xytext=(-4, 8),
                fontsize=8, color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))

    xs = list(main_iterates)
    ys = [f(x) for x in xs]
    ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=1.6, zorder=5)
    for i in range(len(xs) - 1):
        ax.annotate("",
            xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color=st["color"],
                            lw=1.2, mutation_scale=10), zorder=6)
    ax.scatter(xs, ys, color=st["color"], s=40, zorder=7)
    if xs:
        ax.annotate("$x_0=0.5$", (xs[0], ys[0]),
            textcoords="offset points", xytext=(-4, 8),
            fontsize=9, color=st["color"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.8))

    ax.axvline(0, color="0.6", ls=":", lw=1, label="$x^*=0$")
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$f(x)$", fontsize=11)
    ax.set_title(f"{st['label']} — optimization paths", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"manual_example_oscillating_{name.lower()}_path.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def plot_convergence_single(f, name, iterates):
    """f(x_k) vs k on semilogy scale for one method."""
    st = METHOD_STYLE[name]
    ys = [f(x) for x in iterates]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(range(len(ys)), ys,
                color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=6, lw=1.6, label=st["label"])
    ax.set_xlabel("Iteration $k$", fontsize=11)
    ax.set_ylabel("$f(x_k)$", fontsize=11)
    ax.set_title(f"{st['label']} — convergence", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"manual_example_{name.lower()}_convergence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    x0    = np.array([np.pi])
    sigma = 1.0
    f, grad, hess = make_convex(mu=1.0)

    print("\n" + "=" * 60)
    print("Manual examples  —  f(x) = 0.5 x² + 1 − cos x,  x₀ = π")
    print("=" * 60)

    crn_rows,  crn_its,  crn_xf  = run_crn(f, grad, hess, x0, sigma)
    arc_rows,  arc_its,  arc_xf  = run_arc(f, grad, hess, x0, 0.05)
    acrn_rows, acrn_its, acrn_xf = run_acrn(f, grad, hess, x0, sigma)

    crn_cols  = ["k", "x_k", "σ_k", "g_k", "H_k", "h_k", "f(x_k)", "x_{k+1}", "accept"]
    arc_cols  = ["k", "x_k", "σ_k", "h_k", "f(x_k)", "f(x_k+h_k)", "Δm_k", "ρ_k", "accept"]
    acrn_cols = ["k", "x_k", "f(x_k)", "f'(x_k)", "f''(x_k)", "y_k", "h_k", "x_{k+1}"]

    print_table(
        f"NCR  —  f(x)=0.5x²+1−cos(x),  σ₀={sigma},  x₀=π,  x*={crn_xf:.5f},  f*={f(np.array([crn_xf])):.2e}",
        crn_cols, crn_rows,
    )
    print("\nLaTeX:")
    print_latex_table(crn_cols, crn_rows)

    print_table(
        f"ARC  —  f(x)=0.5x²+1−cos(x),  σ₀=0.05,  η₁=0.1,  η₂=0.9,  γ₁=2,  γ₂=4,  x₀=π,  x*={arc_xf:.5f},  f*={f(np.array([arc_xf])):.2e}",
        arc_cols, arc_rows,
    )
    print("\nLaTeX:")
    print_latex_table(arc_cols, arc_rows)

    print_table(
        f"ACRN  —  f(x)=0.5x²+1−cos(x),  L₃={sigma},  σ₀=L₃,  x₀=π,  x*={acrn_xf:.5f},  f*={f(np.array([acrn_xf])):.2e}",
        acrn_cols, acrn_rows,
    )
    print("\nLaTeX:")
    print_latex_table(acrn_cols, acrn_rows)

    results = {
        "NCR":  (crn_rows,  crn_its,  crn_xf),
        "ARC":  (arc_rows,  arc_its,  arc_xf),
        "ACRN": (acrn_rows, acrn_its, acrn_xf),
    }
    print("\nFigures →")
    for name, (_, iterates, _) in results.items():
        plot_path_single(f, name, iterates)
        plot_convergence_single(f, name, iterates)

    # ── oscillating function: 10 + x² - cos(2πx) ─────────────────────────────
    x0_osc    = np.array([0.5])
    sigma_osc = 8 * np.pi**3   # = L3, the Hessian Lipschitz constant of f(x)=10+x²-cos(2πx)
    f2, grad2, hess2 = make_oscillating()

    print("\n" + "=" * 60)
    print("Manual examples  —  f(x) = 10 + x² − cos(2πx),  x₀ = 0.5")
    print("(non-convex: ACRN skipped; σ₀ = 200 ≥ L₃/2 ≈ 124)")
    print("=" * 60)

    crn_rows2,  crn_its2,  crn_xf2  = run_crn(f2, grad2, hess2, x0_osc, sigma_osc)
    arc_rows2,  arc_its2,  arc_xf2  = run_arc(f2, grad2, hess2, x0_osc, 40.0)

    print_table(
        f"NCR  —  f(x)=10+x²−cos(2πx),  σ₀={sigma_osc},  x₀=0.5,  x*={crn_xf2:.5f},  f*={f2(np.array([crn_xf2])):.5f}",
        crn_cols, crn_rows2,
    )
    print("\nLaTeX:")
    print_latex_table(crn_cols, crn_rows2)

    print_table(
        f"ARC  —  f(x)=10+x²−cos(2πx),  σ₀=40,  η₁=0.1,  η₂=0.9,  γ₁=2,  γ₂=4,  x₀=0.5,  x*={arc_xf2:.5f},  f*={f2(np.array([arc_xf2])):.5f}",
        arc_cols, arc_rows2,
    )
    print("\nLaTeX:")
    print_latex_table(arc_cols, arc_rows2)

    _, crn_its_15, _ = run_crn(f2, grad2, hess2, np.array([1.5]), sigma_osc)
    _, crn_its_25, _ = run_crn(f2, grad2, hess2, np.array([2]), sigma_osc)
    _, arc_its_15, _ = run_arc(f2, grad2, hess2, np.array([1.5]), 40.0)
    _, arc_its_25, _ = run_arc(f2, grad2, hess2, np.array([2]), 40.0)

    print("\nFigures →")
    plot_path_oscillating(f2, "NCR", crn_its2,
        [(crn_its_15, "1.5"), (crn_its_25, "2.0")])
    plot_convergence_single(f2, "NCR", crn_its2)
    plot_path_oscillating(f2, "ARC", arc_its2,
        [(arc_its_15, "1.5"), (arc_its_25, "2.0")])


if __name__ == "__main__":
    main()
