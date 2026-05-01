"""
Benchmark tables for thesis.

Table 1 – Convex / strongly convex problems
  Problems : Quadratic, Log-Sum-Exp, Strongly convex poly. (quartic)
  Methods  : Newton, CRN, ACRN, ARC

Table 2 – Nonconvex problems
  Problems : Cubic polynomial, Rosenbrock-2, Rosenbrock-10
  Methods  : Newton, CRN, ARC  (ACRN only when convex at x0)

Columns: Problem | Method | Iter | Unsuccessful | #∇f | Time(s) | ‖∇f‖
"""

import os, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC  = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from methods.NCR  import CubicNewton, CRNOptions
from methods.ARC  import AdaptiveCubicNewton, ARCParams
from methods.ACRN import AcceleratedCubicNewton
from test_functions.quadratic      import quadratic
from test_functions.logsumexp      import logsumexp
from test_functions.quartic_convex import quartic_convex
from test_functions.cubic_poly     import cubic_poly
from test_functions.rosenbrock     import rosenbrock
from utilities import estimate_L3

TOL_GRAD = 1e-6
MAX_ITER = 300

# ── Evaluation counter wrapper ─────────────────────────────────────────────────

import math

def fmt(val):
    """
    Format a float for LaTeX output.
    - Zero              → $0.0000$
    - |exp| <= 3        → $X.XXXX$   (plain fixed 4 d.p.)
    - otherwise         → $M.MMMM_{exp}$  (mantissa with subscript exponent)
    """
    if not isinstance(val, float):
        return str(val)
    if val == 0.0:
        return "$0.0000$"
    exp = math.floor(math.log10(abs(val)))
    if -4 <= exp <= 3:
        return f"${val:.4f}$"
    mantissa = val / 10**exp
    return f"${mantissa:.4f}_{{{exp}}}$"


class _Counted:
    """Wraps a callable and counts calls."""
    def __init__(self, fn):
        self._fn = fn
        self.n = 0
    def __call__(self, *a, **kw):
        self.n += 1
        return self._fn(*a, **kw)
    def reset(self):
        self.n = 0


# ── Standard Newton with Armijo backtracking ───────────────────────────────────

class NewtonOptimizer:
    def __init__(self, f, grad, hess, tol_grad=TOL_GRAD, max_iter=MAX_ITER):
        self.f = f; self.grad = grad; self.hess = hess
        self.tol_grad = tol_grad; self.max_iter = max_iter
        self.log = []; self.runtime = None; self.termination_reason = None

    def run(self, x0):
        x = np.asarray(x0, dtype=float).copy()
        t0 = time.perf_counter()
        for i in range(self.max_iter):
            g = self.grad(x); gnorm = float(np.linalg.norm(g))
            H = np.asarray(self.hess(x), dtype=float); H = 0.5*(H+H.T)
            self.log.append({"iter": i, "f": float(self.f(x)),
                             "grad_norm": gnorm, "accepted": True, "x": x.copy()})
            if gnorm <= self.tol_grad:
                self.termination_reason = "grad_tol"; break
            try:
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                d = np.linalg.solve(H + 1e-8*np.linalg.norm(H,np.inf)*np.eye(len(x)), -g)
            if float(g @ d) >= 0:
                d = -g
            alpha, f0, slope = 1.0, float(self.f(x)), float(g @ d)
            for _ in range(50):
                if float(self.f(x + alpha*d)) <= f0 + 1e-4*alpha*slope:
                    break
                alpha *= 0.5
            x = x + alpha*d
        else:
            self.termination_reason = "max_iter"
        self.runtime = time.perf_counter() - t0
        return x


# ── Convexity check ────────────────────────────────────────────────────────────

def _is_convex_at(hess_fn, x, tol=-1e-6):
    H = np.asarray(hess_fn(x), dtype=float)
    return float(np.linalg.eigvalsh(0.5*(H+H.T)).min()) >= tol


# ── Core runner ────────────────────────────────────────────────────────────────

def run_one(method_name, solver_factory, f_raw, grad_raw, hess_raw, x0):
    """Run a single solver; return stats dict."""
    f_c    = _Counted(f_raw)
    grad_c = _Counted(grad_raw)
    hess_c = _Counted(hess_raw)

    solver = solver_factory(f_c, grad_c, hess_c)
    t0 = time.perf_counter()
    result = solver.run(x0.copy())
    elapsed = time.perf_counter() - t0

    if isinstance(result, tuple):
        x_final, info = result
    else:
        x_final, info = result, {}

    # Prefer solver's own runtime if recorded
    if getattr(solver, "runtime", None) is not None:
        elapsed = solver.runtime

    # Iteration counts from log or info dict
    if hasattr(solver, "log") and solver.log:
        iters       = len(solver.log)
        unsuccessful = sum(1 for e in solver.log if not e.get("accepted", True))
    else:
        iters        = info.get("iterations", 0)
        unsuccessful = 0

    reason    = getattr(solver, "termination_reason", None) or info.get("note", "max_iter")
    g_norm    = float(np.linalg.norm(grad_raw(x_final)))
    f_val     = float(f_raw(x_final))
    success   = (reason == "grad_tol") or (g_norm <= TOL_GRAD)

    return {
        "Method":       method_name,
        "Iter":         iters,
        "Unsuccessful": unsuccessful,
        "#∇f":          grad_c.n,
        "Time (s)":     fmt(elapsed),
        "‖∇f‖":         fmt(g_norm),
    }


def run_problem(problem_name, f_raw, grad_raw, hess_raw, x0,
                methods, sigma=1.0, arc_m0_scale=1.0, L3_known=None):
    """Run all methods on one problem; return list of row dicts."""
    rows = []
    convex_at_x0 = _is_convex_at(hess_raw, x0)
    sigma0 = max(float(sigma), 1e-15)

    for method_name in methods:
        if method_name == "ACRN" and not convex_at_x0:
            rows.append({
                "Problem": problem_name, "Method": "ACRN",
                "Iter": "—", "Unsuccessful": "—",
                "#∇f": "—", "Time (s)": "—", "‖∇f‖": "—",
            })
            continue

        def factory(f, g, h, mn=method_name, L3k=L3_known,
                    s0=sigma0, scale=arc_m0_scale):
            if mn == "Newton":
                return NewtonOptimizer(f, g, h, tol_grad=TOL_GRAD, max_iter=MAX_ITER)
            if mn == "CRN":
                return CubicNewton(f, g, h, options=CRNOptions(
                    sigma0=s0, sigma_min=s0, tol_grad=TOL_GRAD,
                    tol_step=1e-9, max_iter=MAX_ITER))
            if mn == "ARC":
                return AdaptiveCubicNewton(f, g, h, params=ARCParams(
                    sigma0=s0 * scale, sigma_min=s0,
                    eta1=0.1, eta2=0.9, gamma1=2.0, gamma2=4.0,
                    tol_grad=TOL_GRAD, max_iter=MAX_ITER),
                    step_method="secular")
            if mn == "ACRN":
                L3 = L3k if L3k is not None else estimate_L3(h, x0)
                return AcceleratedCubicNewton(f, g, h,
                    L3=max(L3, 1e-15),
                    sigma=max(2.0*L3, 1e-15),  # guard L3=0 (e.g. quadratic)
                    tol_grad=TOL_GRAD, tol_step=1e-12,
                    max_iter=MAX_ITER, adaptive_sigma=(L3k is None and L3 > 1e-14))
            raise ValueError(mn)

        row = run_one(method_name, factory, f_raw, grad_raw, hess_raw, x0)
        row["Problem"] = problem_name
        rows.append(row)
    return rows


# ── Problem definitions (mirroring visualize_2d.py exactly) ───────────────────

# Quadratic  — A=[[4,1],[1,3]], b=[1,2], sigma=0, x0=[2,-1.5]
A_quad = np.array([[4.0, 1.0], [1.0, 3.0]])
b_quad = np.array([1.0, 2.0])
f_quad, g_quad, h_quad, L3_quad = quadratic(A_quad, b_quad)
x0_quad = np.array([2.0, -1.5])

# Log-Sum-Exp  — a=[[1,0],[0,1],[-1,-1]], b=0, mu=0.5, arc_m0_scale=10
a_lse = np.array([[1, 0], [0, 1], [-1, -1]], dtype=float)
b_lse = np.zeros(3)
f_lse, g_lse, h_lse, M_lse = logsumexp(a_lse, b_lse, mu=0.5)
x0_lse = np.array([2.0, -1.5])

# Quartic convex  — n=2, sigma=2.0, x0=[2,-1.5]
f_qrt, g_qrt, h_qrt = quartic_convex(n=2)
x0_qrt = np.array([2.0, -1.5])

# Cubic poly  — Q=[[2,.5],[.5,1]], gamma=1, sigma=L3=2, x0=[2,-1.5]
Q_cp = np.array([[2.0, 0.5], [0.5, 1.0]])
f_cp, g_cp, h_cp, L3_cp = cubic_poly(Q_cp, gamma=1.0)
x0_cp = np.array([2.0, -1.5])

# Rosenbrock-2  — sigma=1, x0=[0,0]
f_rb2, g_rb2, h_rb2 = rosenbrock(N=2)
x0_rb2 = np.zeros(2)

# Rosenbrock-10  — sigma=1, x0=zeros(10)
f_rb10, g_rb10, h_rb10 = rosenbrock(N=10)
x0_rb10 = np.zeros(10)


# ── Run all problems ───────────────────────────────────────────────────────────

CONVEX_METHODS    = ["Newton", "CRN", "ACRN", "ARC"]
NONCONVEX_METHODS = ["Newton", "CRN", "ARC", "ACRN"]  # ACRN skipped if non-convex

convex_rows = []
convex_rows += run_problem("Quadratic",            f_quad, g_quad, h_quad, x0_quad,
                           CONVEX_METHODS, sigma=0.0, arc_m0_scale=1.0, L3_known=L3_quad)
convex_rows += run_problem("Log-Sum-Exp",          f_lse,  g_lse,  h_lse,  x0_lse,
                           CONVEX_METHODS, sigma=float(M_lse), arc_m0_scale=10.0)
convex_rows += run_problem("Strongly convex poly.", f_qrt,  g_qrt,  h_qrt,  x0_qrt,
                           CONVEX_METHODS, sigma=2.0,          arc_m0_scale=1.0)

nonconvex_rows = []
nonconvex_rows += run_problem("Cubic polynomial",  f_cp,  g_cp,  h_cp,  x0_cp,
                              NONCONVEX_METHODS, sigma=L3_cp, arc_m0_scale=1.0, L3_known=L3_cp)
nonconvex_rows += run_problem("Rosenbrock-2",      f_rb2,  g_rb2,  h_rb2,  x0_rb2,
                              NONCONVEX_METHODS, sigma=1.0, arc_m0_scale=1.0)
nonconvex_rows += run_problem("Rosenbrock-10",     f_rb10, g_rb10, h_rb10, x0_rb10,
                              NONCONVEX_METHODS, sigma=1.0, arc_m0_scale=1.0)


# ── Build DataFrames ───────────────────────────────────────────────────────────

COLS = ["Problem", "Method", "Iter", "Unsuccessful", "#∇f", "Time (s)", "‖∇f‖"]

df_convex    = pd.DataFrame(convex_rows,    columns=COLS)
df_nonconvex = pd.DataFrame(nonconvex_rows, columns=COLS)


# ── Print tables ───────────────────────────────────────────────────────────────

def _print_table(df, title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)
    print(df.to_string(index=False))

_print_table(df_convex,    "Table 1 — Convex / Strongly Convex Problems")
_print_table(df_nonconvex, "Table 2 — Nonconvex Problems")


# ── LaTeX output ───────────────────────────────────────────────────────────────

def _to_latex(df, label, caption):
    # Group by Problem with multirow-style blank repetition
    prev = None
    rows_tex = []
    for _, row in df.iterrows():
        prob = row["Problem"] if row["Problem"] != prev else ""
        prev = row["Problem"]
        vals = [prob] + [str(row[c]) for c in COLS[1:]]
        rows_tex.append(" & ".join(vals) + r" \\")

    col_fmt = "ll" + "r" * (len(COLS) - 2)
    header  = " & ".join(COLS) + r" \\"
    hlines  = []
    prev = None
    line_idx = 0
    for _, row in df.iterrows():
        if row["Problem"] != prev and prev is not None:
            hlines.append(line_idx)
        prev = row["Problem"]
        line_idx += 1

    body_lines = []
    for i, line in enumerate(rows_tex):
        if i in hlines:
            body_lines.append(r"    \hline")
        body_lines.append(f"    {line}")

    tex = (
        r"\begin{table}[ht]" + "\n"
        r"  \centering" + "\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        r"  \begin{tabular}{" + col_fmt + "}\n"
        r"    \toprule" + "\n"
        f"    {header}\n"
        r"    \midrule" + "\n"
        + "\n".join(body_lines) + "\n"
        + r"    \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}"
    )
    return tex

tex_convex = _to_latex(
    df_convex,
    label="tab:convex-results",
    caption="Benchmark results on convex and strongly convex problems."
)
tex_nonconvex = _to_latex(
    df_nonconvex,
    label="tab:nonconvex-results",
    caption="Benchmark results on nonconvex problems."
)

print("\n\n% ── LaTeX Table 1 ────────────────────────────────────────────────────")
print(tex_convex)
print("\n% ── LaTeX Table 2 ────────────────────────────────────────────────────")
print(tex_nonconvex)


# ── Save CSVs ─────────────────────────────────────────────────────────────────

out_dir = os.path.dirname(__file__)
df_convex.to_csv(   os.path.join(out_dir, "results_convex.csv"),    index=False)
df_nonconvex.to_csv(os.path.join(out_dir, "results_nonconvex.csv"), index=False)
print(f"\nSaved: results_convex.csv, results_nonconvex.csv → {out_dir}")
