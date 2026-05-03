# Newton-type Cubic Regularization Methods

A thesis project implementing and comparing three second-order optimization algorithms based on cubic regularization: **NCR**, **ARC**, and **ACRN**. The goal is to study how cubic regularization improves upon classical Newton's method in terms of convergence guarantees, robustness, and practical performance — across both convex and non-convex problems.

## Background

Classical Newton's method minimizes a quadratic model of the objective at each step. While it achieves fast local convergence, it lacks global convergence guarantees and can fail when the Hessian is indefinite. Cubic regularization addresses this by adding a cubic penalty term to the model:

```
m(h) = f(x) + g^T h + 1/2 h^T H h + (σ/3)||h||³
```

The cubic term bounds the error of the quadratic approximation, ensuring the model is always a valid upper bound (under Lipschitz-Hessian assumptions) and guaranteeing descent. This leads to provably better worst-case complexity than gradient descent or basic Newton steps, without sacrificing the fast local convergence of Newton-type methods.

This repository implements three variants of this idea and compares them empirically across a range of test functions, dimensions, and problem conditions.

## Methods

| Method | Full name | Convergence rate | Notes |
|--------|-----------|-----------------|-------|
| **NCR** | Cubic-Regularized Newton | O(k⁻²) in gradient norm | Nesterov & Polyak (2006). σ is halved on success, doubled on failure. |
| **ARC** | Adaptive Regularisation with Cubics | O(k⁻²) in gradient norm | Cartis, Gould & Toint (2011). Self-tuning σ via agreement ratio ρ = actual/predicted reduction. Often faster in practice. |
| **ACRN** | Accelerated Cubic-Regularized Newton | O(k⁻³) in function value | Nesterov (2008). Applies estimate-sequence acceleration on top of NCR. Requires convexity and a known Hessian Lipschitz constant L3. |
| **Newton** | Pure Newton (baseline) | Quadratic locally | No regularization. Included as a reference; can fail on indefinite Hessians. |

All methods solve the cubic subproblem globally via bisection on the secular equation, guaranteeing the true minimizer of the cubic model at each step.

## Key references

- Y. Nesterov and B. Polyak. "Cubic regularization of Newton method and its global performance". In: *Mathematical Programming* 108 (2006), pp. 177–205. doi: [10.1007/s10107-006-0706-8](https://doi.org/10.1007/s10107-006-0706-8).
- Y. Nesterov. "Accelerating the cubic regularization of Newton's method on convex problems". In: *Mathematical Programming* 112 (2007), pp. 159–181. doi: [10.1007/s10107-006-0089-x](https://doi.org/10.1007/s10107-006-0089-x).
- C. Cartis, N.I.M. Gould, and P.L. Toint. "Adaptive cubic regularisation methods for unconstrained optimization. Part I: motivation, convergence and numerical results". In: *Mathematical Programming* 127 (2009), pp. 245–295. doi: [10.1007/s10107-009-0286-5](https://doi.org/10.1007/s10107-009-0286-5).
- C. Cartis, N.I.M. Gould, and P.L. Toint. "Adaptive cubic regularisation methods for unconstrained optimization. Part II: worst-case function- and derivative-evaluation complexity". In: *Mathematical Programming* 130 (2010), pp. 295–319. doi: [10.1007/s10107-009-0337-y](https://doi.org/10.1007/s10107-009-0337-y).

## Setup

Requires Python 3.11+ and must be run inside WSL/Ubuntu (pycutest depends on Linux).

```bash
source venv/bin/activate
pip install -e .          # installs src/ as editable package "newton-methods"
```

Key dependencies (tested versions):

| Package | Version |
|---------|---------|
| Python | 3.12.3 |
| numpy | 2.4.3 |
| scipy | 1.17.1 |
| matplotlib | 3.10.8 |
| pandas | 3.0.1 |
| pycutest | 1.8.0 |

## Running experiments

| Script | What it does | Output |
|---|---|---|
| `experiments/0_stopping_criteria_sensitivity/run_experiment.py` | Stopping criterion sensitivity across solvers and tolerances | `results/0_stopping_criteria_sensitivity/` |
| `experiments/1_initial_point_sensitivity/run_experiment.py` | Effect of initial point on convergence | `results/1_initial_point_sensitivity/` |
| `experiments/2_robustness_methods/run_robustness_methods.py` | Solver robustness across problem families | `results/2_robustness_methods/` |
| `experiments/3_sensitivity_sigma/run_sigma_sweep.py` | Sensitivity to initial regularization σ₀ | `results/3_sensitivity_sigma/` |
| `experiments/4_testing/benchmark.py` | Main solver comparison table (convex + non-convex) | `results/4_testing/results.csv` |
| `experiments/5_large_scale_synthesis/run_synthesis.py` | Scaling behaviour across dimensions | `results/5_large_scale_synthesis/` |
| `experiments/6_usefull/abc.py` | CUTEst benchmark (Appendix A) | `results/6_usefull/` |

All scripts are run from the repo root with the venv active.

## Project structure

```
src/
├── methods/
│   ├── NCR.py            # Cubic-Regularized Newton
│   ├── ARC.py            # Adaptive Regularisation with Cubics
│   ├── ACRN.py           # Accelerated Cubic-Regularized Newton
│   └── pure_newton.py    # Pure Newton (reference baseline)
├── test_functions/
│   ├── quadratic.py
│   ├── cubic_norm.py
│   ├── quartic_convex.py
│   ├── logsumexp.py
│   ├── rosenbrock.py
│   ├── dixon_price.py
│   └── rastrigin.py
└── utilities.py          # estimate_L3, is_convex_at, summarize

experiments/             # numbered experiment scripts (0–6)
results/                  # all output written here (CSV + figures/)
tests/                    # manual verification scripts
```

## Solver interface

Every solver follows the same pattern:

```python
solver = SolverClass(f, grad, hess, **kwargs)
x_star = solver.run(x0)

solver.log                 # list[dict] — one entry per iteration
solver.runtime             # float — wall-clock seconds
solver.termination_reason  # str — "grad_tol" | "step_tol" | "max_iter"
```

Each log entry contains: `iter`, `f`, `g`, `H`, `grad_norm`, `step_norm`, `h`, `accepted`, `sigma`, `x`.

ACRN returns a `(x, info)` tuple instead of just `x`; `runtime` is not stored — time externally if needed.

## Implementation notes

- Hessians are symmetrized as `0.5*(H + H.T)` before use.
- Singular linear systems get a diagonal jitter of `1e-12 * ||A||_inf` for numerical stability.
- ACRN is only valid for convex problems. Benchmark scripts verify `min_eig(H(x0)) >= -1e-6` before running it.
- When L3 (Hessian Lipschitz constant) is not known analytically, it is estimated via finite differences in random directions from x0 using `utilities.estimate_L3`.
- The cubic subproblem `min g^T h + 0.5 h^T H h + (σ/3)||h||^3` is solved via bisection on the secular equation.
