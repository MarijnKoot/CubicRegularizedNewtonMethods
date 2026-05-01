# Robustness of Methods w.r.t. Problem Geometry

Generated: 2026-05-01 02:34:54

## Definition of Robustness

A method is considered **robust** if it consistently produces reliable, stable, and
high-quality solutions when the problem geometry becomes difficult.
Concretely, robustness is measured by:
- **Success rate**: fraction of runs that satisfy ‖∇f‖ ≤ ε_g = 1e-7
- **Iteration count**: median iterations over the set of starting points
- **Numerical stability**: absence of NaN/Inf in iterates
- **Sensitivity ratio**: max/min iterations across starts (lower = more robust)

## Benchmark Families

| Family | Problems | Geometric difficulty |
|--------|----------|----------------------|
| Ill-conditioned quadratics | SPD quadratics with κ ∈ {1e2, 1e4, 1e6, 1e8} + near-singular | Numerical stability, Newton breakdown |
| Rosenbrock | n = 2, 10, 20 | Narrow curved valley, slow convergence |
| Dixon-Price | n = 10, 20, 50 | Coupled variable interactions |
| Rastrigin | n = 10, 20, 50 | Highly multimodal, many local minima |

## Protocol

- `eps_g = 1e-7` (gradient-norm stopping), `max_iter = 500`
- 2 starting points per instance (standard + one variant)
- ACRN only applied to convex problems
- Seed = 42 throughout

## Results Summary

### Ill-conditioned quadratics

| Problem | Method | Success | Med. iters | Med. runtime (s) | Rejected (med) |
|---------|--------|---------|------------|------------------|----------------|
| Quadratic $\kappa$=1e+06 | Newton | 10/10 | 1 | 0.0003 | 0 |
| Quadratic $\kappa$=1e+06 | NCR | 10/10 | 5 | 0.0097 | 0 |
| Quadratic $\kappa$=1e+06 | ARC | 10/10 | 5 | 0.0071 | 0 |
| Quadratic $\kappa$=1e+06 | ACRN | 10/10 | 1 | 0.0025 | 0 |
| NearPSD $\lambda_{\min}$=1e-06 | Newton | 10/10 | 1 | 0.0001 | 0 |
| NearPSD $\lambda_{\min}$=1e-06 | NCR | 10/10 | 18 | 0.0247 | 0 |
| NearPSD $\lambda_{\min}$=1e-06 | ARC | 10/10 | 18 | 0.0217 | 0 |
| NearPSD $\lambda_{\min}$=1e-06 | ACRN | 10/10 | 1 | 0.0013 | 0 |
| NearPSD (singular) | Newton | 10/10 | 1 | 0.0001 | 0 |
| NearPSD (singular) | NCR | 10/10 | 7 | 0.0081 | 0 |
| NearPSD (singular) | ARC | 10/10 | 7 | 0.0086 | 0 |
| NearPSD (singular) | ACRN | 10/10 | 1 | 0.0016 | 0 |
| Quadratic $\kappa$=1e+04 | Newton | 10/10 | 1 | 0.0001 | 0 |
| Quadratic $\kappa$=1e+04 | NCR | 10/10 | 5 | 0.0066 | 0 |
| Quadratic $\kappa$=1e+04 | ARC | 10/10 | 5 | 0.0092 | 0 |
| Quadratic $\kappa$=1e+04 | ACRN | 10/10 | 1 | 0.0039 | 0 |
| Quadratic $\kappa$=1e+08 | Newton | 10/10 | 1 | 0.0001 | 0 |
| Quadratic $\kappa$=1e+08 | NCR | 10/10 | 5 | 0.0056 | 0 |
| Quadratic $\kappa$=1e+08 | ARC | 10/10 | 5 | 0.0064 | 0 |
| Quadratic $\kappa$=1e+08 | ACRN | 10/10 | 1 | 0.0082 | 3 |
| Quadratic $\kappa$=1e+02 | Newton | 10/10 | 1 | 0.0001 | 0 |
| Quadratic $\kappa$=1e+02 | NCR | 10/10 | 5 | 0.0051 | 0 |
| Quadratic $\kappa$=1e+02 | ARC | 10/10 | 5 | 0.0049 | 0 |
| Quadratic $\kappa$=1e+02 | ACRN | 10/10 | 1 | 0.0011 | 0 |
| NearPSD $\lambda_{\min}$=1e-04 | Newton | 10/10 | 1 | 0.0001 | 0 |
| NearPSD $\lambda_{\min}$=1e-04 | NCR | 10/10 | 14 | 0.0184 | 0 |
| NearPSD $\lambda_{\min}$=1e-04 | ARC | 10/10 | 14 | 0.0157 | 0 |
| NearPSD $\lambda_{\min}$=1e-04 | ACRN | 10/10 | 1 | 0.0012 | 0 |

### Rosenbrock

| Problem | Method | Success | Med. iters | Med. runtime (s) | Rejected (med) |
|---------|--------|---------|------------|------------------|----------------|
| Rosenbrock $n$=2 | Newton | 10/10 | 5 | 0.0008 | 0 |
| Rosenbrock $n$=2 | NCR | 10/10 | 30 | 0.0418 | 12 |
| Rosenbrock $n$=2 | ARC | 10/10 | 22 | 0.0316 | 4 |
| Rosenbrock $n$=10 | Newton | 10/10 | 35 | 0.0063 | 0 |
| Rosenbrock $n$=10 | NCR | 10/10 | 52 | 0.0853 | 22 |
| Rosenbrock $n$=10 | ARC | 10/10 | 38 | 0.0621 | 8 |
| Rosenbrock $n$=20 | Newton | 9/10 | 49 | 0.0129 | 0 |
| Rosenbrock $n$=20 | NCR | 10/10 | 85 | 0.1885 | 40 |
| Rosenbrock $n$=20 | ARC | 10/10 | 56 | 0.1067 | 13 |

### Dixon-Price

| Problem | Method | Success | Med. iters | Med. runtime (s) | Rejected (med) |
|---------|--------|---------|------------|------------------|----------------|
| Dixon-Price $n$=10 | Newton | 10/10 | 22 | 0.0043 | 0 |
| Dixon-Price $n$=10 | NCR | 10/10 | 18 | 0.0235 | 8 |
| Dixon-Price $n$=10 | ARC | 10/10 | 14 | 0.0287 | 4 |
| Dixon-Price $n$=20 | Newton | 10/10 | 38 | 0.0097 | 0 |
| Dixon-Price $n$=20 | NCR | 10/10 | 42 | 0.0787 | 20 |
| Dixon-Price $n$=20 | ARC | 10/10 | 28 | 0.0553 | 8 |
| Dixon-Price $n$=50 | Newton | 7/10 | 83 | 0.0392 | 0 |
| Dixon-Price $n$=50 | NCR | 10/10 | 52 | 0.1918 | 24 |
| Dixon-Price $n$=50 | ARC | 10/10 | 36 | 0.1534 | 10 |

### Rastrigin

| Problem | Method | Success | Med. iters | Med. runtime (s) | Rejected (med) |
|---------|--------|---------|------------|------------------|----------------|
| Rastrigin $n$=10 | Newton | 9/10 | 7 | 0.0006 | 0 |
| Rastrigin $n$=10 | NCR | 9/10 | 16 | 0.0522 | 11 |
| Rastrigin $n$=10 | ARC | 10/10 | 10 | 0.0288 | 5 |
| Rastrigin $n$=20 | Newton | 7/10 | 7 | 0.0013 | 0 |
| Rastrigin $n$=20 | NCR | 8/10 | 17 | 0.0957 | 12 |
| Rastrigin $n$=20 | ARC | 6/10 | 11 | 0.0663 | 14 |
| Rastrigin $n$=50 | Newton | 7/10 | 7 | 0.0017 | 0 |
| Rastrigin $n$=50 | NCR | 9/10 | 17 | 0.2180 | 11 |
| Rastrigin $n$=50 | ARC | 8/10 | 12 | 0.1847 | 5 |

## Output Files

| Path | Description |
|------|-------------|
| `raw/raw_results.csv` | One row per (instance, method, start) |
| `summary/aggregated.csv` | Grouped statistics |
| `summary/latex_tables.txt` | LaTeX booktabs tables per family |
| `figures/ill_cond_*` | Ill-conditioned quadratic plots |
| `figures/rosenbrock2_paths.png` | Path plots for Rosenbrock n=2 |
| `figures/{family}_*_by_dim.png` | Bar plots by dimension |
| `figures/rejected_steps_by_family.png` | Rejected steps summary |
| `figures/success_rate_heatmap.png` | Overview heatmap |
