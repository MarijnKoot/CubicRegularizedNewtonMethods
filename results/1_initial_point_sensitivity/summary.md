# Initial Point Sensitivity

Generated: 2026-05-01 02:33:51

## Problems

- **LogSumExp**: LogSumExp (n=10) — convex, random (a,b) with seed 42
- **QuarticConvex**: QuarticConvex (n=10) — strongly convex, f(x)=0.5||x||²+0.25||x||⁴
- **Rosenbrock**: Rosenbrock (n=10) — nonconvex, global min at x=(1,...,1)

## Starting Points

| Index | Label | Description |
|-------|-------|-------------|
| 0 | standard | Problem-specific fixed start (e.g. x=(-1,...,-1) for Rosenbrock) |
| 1 | benign | Closer to known minimizer |
| 2–6 | random_i | standard + N(0, 2.0²) noise, seed 42 |

## Protocol

- `eps_g = 1e-07`, `max_iter = 500`, `dim = 10`, `seed = 42`
- ACRN skipped for nonconvex problems

## Aggregated Results

| Problem | Method | Success | Med. Iters | IQR | Sensitivity ratio |
|---------|--------|---------|------------|-----|-------------------|
| logsumexp | ACRN | 2/7 | 501 | 0 | 3.1 |
| logsumexp | ARC | 7/7 | 10 | 1 | 2.2 |
| logsumexp | NCR | 7/7 | 10 | 2 | 2.4 |
| logsumexp | Newton | 1/7 | 500 | 0 | — |
| quartic_convex | ACRN | 1/7 | 501 | 0 | — |
| quartic_convex | ARC | 7/7 | 10 | 0 | 2.5 |
| quartic_convex | NCR | 7/7 | 10 | 0 | 2.5 |
| quartic_convex | Newton | 7/7 | 10 | 0 | 2.5 |
| rosenbrock | ACRN | 0/7 | — | — | — |
| rosenbrock | ARC | 7/7 | 36 | 6 | 2.8 |
| rosenbrock | NCR | 7/7 | 51 | 6 | 2.9 |
| rosenbrock | Newton | 7/7 | 37 | 15 | 3.8 |

## Output Files

| File | Description |
|------|-------------|
| `raw_results.csv` | One row per (problem, method, starting point) |
| `aggregated.csv` | Summary stats grouped by (problem, method) |
| `figures/{problem}_iters_boxplot.png` | Iteration count distribution |
| `figures/{problem}_runtime_boxplot.png` | Runtime distribution |
| `figures/{problem}_final_f_boxplot.png` | Final objective distribution |
