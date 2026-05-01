# Large-Scale Synthesis Experiment

Generated: 2026-05-01 02:36:57

## Experimental Defaults

| Setting | Value | Source |
|---------|-------|--------|
| Gradient tolerance (ε_g) | 1e-7 | experiments/2 and 3 |
| Max iterations | 500 | all earlier experiments |
| Seed | 42 | all earlier experiments |
| σ₀ policy | max(2·L3, 0.5) | adapters.py default |
| Methods | ['Newton', 'NCR', 'ARC', 'ACRN'] | |

## Benchmark Families

| Family | Purpose | Dimensions |
|--------|---------|------------|
| LogSumExp | Convex scalability; fair CRN vs ACRN | 10, 20, 50, 100 |
| Rosenbrock | Nonconvex scalability; narrow valley | 2, 10, 20, 50, 100 |
| Quadratic (ill-cond.) | Convex, ill-conditioned; reconnect to Newton fragility | n∈{10,50,100}, κ∈{1e2,1e4,1e6} |

## Starting Points

Two starts per instance: `standard` and `benign`.
Rosenbrock: standard = (−1,…,−1), benign = (0.5,…,0.5).
Others: standard = N(0,1) (seed 1042), benign = 0.

## Results Summary

### Logsumexp

| Problem | Method | Rate | Med.k | Med.t(s) |
|---------|--------|------|-------|----------|
| LogSumExp n=10 | Newton | 0.00 | 500 | 0.393 |
| LogSumExp n=10 | NCR | 1.00 | 10 | 0.022 |
| LogSumExp n=10 | ARC | 1.00 | 10 | 0.020 |
| LogSumExp n=10 | ACRN | 0.50 | 448 | 2.403 |
| LogSumExp n=20 | Newton | 0.50 | 254 | 0.179 |
| LogSumExp n=20 | NCR | 1.00 | 14 | 0.041 |
| LogSumExp n=20 | ARC | 1.00 | 8 | 0.031 |
| LogSumExp n=20 | ACRN | 0.00 | 501 | 2.180 |
| LogSumExp n=50 | Newton | 0.00 | 500 | 1.334 |
| LogSumExp n=50 | NCR | 1.00 | 12 | 0.110 |
| LogSumExp n=50 | ARC | 1.00 | 12 | 0.096 |
| LogSumExp n=50 | ACRN | 0.00 | 501 | 14.990 |

### Rosenbrock

| Problem | Method | Rate | Med.k | Med.t(s) |
|---------|--------|------|-------|----------|
| Rosenbrock n=2 | Newton | 1.00 | 5 | 0.002 |
| Rosenbrock n=2 | NCR | 1.00 | 30 | 0.111 |
| Rosenbrock n=2 | ARC | 1.00 | 22 | 0.080 |
| Rosenbrock n=2 | ACRN | 0.00 | 501 | 3.163 |
| Rosenbrock n=10 | Newton | 1.00 | 32 | 0.005 |
| Rosenbrock n=10 | NCR | 1.00 | 40 | 0.123 |
| Rosenbrock n=10 | ARC | 1.00 | 28 | 0.088 |
| Rosenbrock n=10 | ACRN | 0.00 | 501 | 3.814 |
| Rosenbrock n=20 | Newton | 1.00 | 160 | 0.060 |
| Rosenbrock n=20 | NCR | 1.00 | 57 | 0.156 |
| Rosenbrock n=20 | ARC | 1.00 | 39 | 0.162 |
| Rosenbrock n=20 | ACRN | 0.00 | 501 | 4.157 |
| Rosenbrock n=50 | Newton | 1.00 | 178 | 0.053 |
| Rosenbrock n=50 | NCR | 1.00 | 102 | 0.765 |
| Rosenbrock n=50 | ARC | 1.00 | 71 | 0.668 |
| Rosenbrock n=50 | ACRN | 0.00 | 501 | 12.739 |

### Ill Conditioned

| Problem | Method | Rate | Med.k | Med.t(s) |
|---------|--------|------|-------|----------|
| Quadratic n=10 κ=1e+04 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=10 κ=1e+04 | NCR | 1.00 | 4 | 0.013 |
| Quadratic n=10 κ=1e+04 | ARC | 1.00 | 4 | 0.013 |
| Quadratic n=10 κ=1e+04 | ACRN | 1.00 | 1 | 0.001 |
| Quadratic n=10 κ=1e+06 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=10 κ=1e+06 | NCR | 1.00 | 4 | 0.014 |
| Quadratic n=10 κ=1e+06 | ARC | 1.00 | 4 | 0.014 |
| Quadratic n=10 κ=1e+06 | ACRN | 1.00 | 1 | 0.001 |
| Quadratic n=10 κ=1e+02 | Newton | 1.00 | 1 | 0.010 |
| Quadratic n=10 κ=1e+02 | NCR | 1.00 | 4 | 0.003 |
| Quadratic n=10 κ=1e+02 | ARC | 1.00 | 4 | 0.013 |
| Quadratic n=10 κ=1e+02 | ACRN | 1.00 | 1 | 0.001 |
| Quadratic n=20 κ=1e+06 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=20 κ=1e+06 | NCR | 1.00 | 4 | 0.004 |
| Quadratic n=20 κ=1e+06 | ARC | 1.00 | 4 | 0.014 |
| Quadratic n=20 κ=1e+06 | ACRN | 1.00 | 1 | 0.012 |
| Quadratic n=20 κ=1e+04 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=20 κ=1e+04 | NCR | 1.00 | 4 | 0.014 |
| Quadratic n=20 κ=1e+04 | ARC | 1.00 | 4 | 0.014 |
| Quadratic n=20 κ=1e+04 | ACRN | 1.00 | 1 | 0.001 |
| Quadratic n=20 κ=1e+02 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=20 κ=1e+02 | NCR | 1.00 | 5 | 0.026 |
| Quadratic n=20 κ=1e+02 | ARC | 1.00 | 5 | 0.015 |
| Quadratic n=20 κ=1e+02 | ACRN | 1.00 | 1 | 0.002 |
| Quadratic n=50 κ=1e+04 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=50 κ=1e+04 | NCR | 1.00 | 5 | 0.044 |
| Quadratic n=50 κ=1e+04 | ARC | 1.00 | 5 | 0.044 |
| Quadratic n=50 κ=1e+04 | ACRN | 1.00 | 1 | 0.014 |
| Quadratic n=50 κ=1e+02 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=50 κ=1e+02 | NCR | 1.00 | 5 | 0.032 |
| Quadratic n=50 κ=1e+02 | ARC | 1.00 | 5 | 0.032 |
| Quadratic n=50 κ=1e+02 | ACRN | 1.00 | 1 | 0.013 |
| Quadratic n=50 κ=1e+06 | Newton | 1.00 | 1 | 0.000 |
| Quadratic n=50 κ=1e+06 | NCR | 1.00 | 5 | 0.035 |
| Quadratic n=50 κ=1e+06 | ARC | 1.00 | 5 | 0.040 |
| Quadratic n=50 κ=1e+06 | ACRN | 1.00 | 1 | 0.013 |

## Output Files

| Path | Description |
|------|-------------|
| `raw/raw_results.csv` | One row per (instance, method, start) |
| `summary/aggregated.csv` | Grouped medians per (instance, method) |
| `latex/table_{family}.tex` | Compact LaTeX tables per family |
| `latex/table_acrn_gain.tex` | ACRN vs CRN iteration/runtime gain (convex) |
| `figures/scaling_{family}.pdf` | Iteration and runtime vs dimension |
