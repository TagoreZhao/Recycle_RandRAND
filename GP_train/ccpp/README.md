# CCPP kernel-ridge PCG benchmark

**Recycling preconditioners across a regularization sweep for Gaussian-process
regression on the UCI Combined Cycle Power Plant dataset.**

The main experiment is [`run_benchmark.m`](run_benchmark.m): it builds one dense
RBF kernel and, as the ridge/noise term `σ²` sweeps from `1e-8` to `1`, compares
how many PCG iterations each of several preconditioners needs — with a focus on
whether deflation / coarse-space bases can be built **once and reused** across the
whole sweep.

---

## Problem

Gaussian-process regression (equivalently, kernel ridge regression) fits a target
`y` from features `X` by solving the symmetric positive-definite (SPD) linear
system

```
A x = b,     A = K + σ²·I
```

where `K` is the RBF (squared-exponential) kernel matrix over the training points,
`σ²` is the observation noise / ridge regularization, and `b` is the standardized
target. Solving this system is the computational bottleneck of GP training.

Two things make it hard:

- **`K` is dense** — memory and per-iteration cost scale as `O(n²)`.
- **The system is ill-conditioned**, and it gets worse as `σ² → 0`: the ridge term
  is what lifts `K`'s tiny eigenvalues away from zero, so shrinking `σ²` inflates
  the condition number and PCG iteration counts blow up.

This benchmark asks two questions:

1. How well do different preconditioners keep PCG iteration counts bounded as `σ²`
   sweeps across eight orders of magnitude?
2. Can the coarse/deflation space be **recycled** across the sweep instead of
   rebuilt at every `σ²`? Because `A = K + σ²·I` shares its eigenvectors with `K`
   for **every** `σ²`, a coarse space built from `K` once is exactly reusable for
   the whole sweep — that reuse is the central idea being studied.

---

## Dataset

[UCI Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294)
(dataset 294), `Folds5x2_pp.xlsx`:

- **9568** hourly records collected from a power plant at full load.
- **4 features** → **1 target**:

  | Column | Meaning |
  |--------|---------|
  | AT | Ambient temperature |
  | V  | Exhaust vacuum |
  | AP | Ambient pressure |
  | RH | Relative humidity |
  | **PE** | **Net hourly electrical energy output** (regression target — last column) |

[`download_ccpp.m`](download_ccpp.m) fetches the UCI zip, reads the spreadsheet,
and writes a header-less numeric `data/ccpp.csv` (last column = target). The
`data/` directory is gitignored — it is reproducible from code and never committed.

`run_benchmark.m` draws a random **`n = 3000`** subset, standardizes each feature
and the target to zero mean / unit variance ([`standardize_data.m`](standardize_data.m)),
and sets the RBF lengthscale from the **median pairwise-distance heuristic**
([`estimate_median_lengthscale.m`](estimate_median_lengthscale.m); `ell ≈ 2.53`).

---

## Method

Build the dense RBF kernel `K` **once** ([`rbf_kernel_matrix.m`](rbf_kernel_matrix.m)):

```
K(i,k) = exp( -||X(i,:) - X(k,:)||² / (2·ell²) )
```

Then for each `σ²` in `logspace(-8, 0, 10)`, solve `A(σ²) x = b` with MATLAB's
`pcg` (tolerance `1e-6`). The system matrix is applied matrix-free as
`x ↦ K·x + σ²·x` — a dense `A` is never formed. Each of the following
preconditioners produces one curve in the output plot:

| Solver | What it is |
|--------|------------|
| `unprec` | Plain PCG, no preconditioner — the baseline. |
| `ichol` | Incomplete Cholesky ([`build_ichol_robust.m`](build_ichol_robust.m), which escalates a diagonal shift when a pivot goes non-positive — common for dense kernels). |
| `amg` | 2-level smoothed-aggregation algebraic multigrid V-cycle, reusing the ichol factor as its fine-level smoother. |
| `defl_P` | **Deflation** preconditioner `P = (I − VV') + τ·V(VᵀAV)⁻¹Vᵀ`, with `V` = the `DEFLAT_LG_EIG = 100` largest eigenvectors of `A` and coarse weight `τ = 0.5`. |
| `defl_sketch_q1/q2/q3` | Same deflation `P`, but the coarse basis is a cheap **Gaussian-sketched power iteration** `Y = K^q·Ω` (`Ω = randn`), trading exactness for construction cost. `q` = number of `K`-applies. |
| `twolevel_VAhat` | Two-level split scheme `B = L⁻ᵀ P L⁻¹` applied on the ichol-split operator `Â = L⁻¹ A L⁻ᵀ`, with `V` = largest eigenvectors of `Â`. |

**Recycling.** With `DEFLAT_PREC_REFRESH = Inf`, the deflation and sketch bases are
built at the first `σ²` and reused for all the rest. This reuse is **mathematically
exact** for the bases derived from `K` (`defl_P`, the sketches — since `A` shares
`K`'s eigenvectors), and **approximate** for `twolevel_VAhat`, whose `Â` depends on
the ichol factor `L` that changes with `σ²`. The ichol and AMG factors are rebuilt
at every `σ²`.

The advanced preconditioners live in the repo-root package
[`+src/+precond/`](../../+src/+precond) (`deflation_P_apply`,
`subspace_iter_plain`, `make_amg_preconditioner`); `run_benchmark.m` adds the repo
root to the path automatically so `src.precond.*` resolves.

---

## Quick start

From the `GP_train/ccpp/` directory in MATLAB:

```matlab
download_ccpp        % one-time: fetch data/ccpp.csv from UCI (no-op if present)
run_benchmark        % run the full sigma^2 sweep -> benchmark_sigma2/
```

For a fast end-to-end check (n = 400, 3 `σ²` values), set the toggle in the base
workspace first:

```matlab
SMOKE_TEST = 1;
run_benchmark        % writes to benchmark_sigma2_smoke/ instead
```

`run_benchmark.m` requires the AMG/deflation helpers under the repo root, which the
driver puts on the path for you.

---

## Outputs

Written to `benchmark_sigma2/` (or `benchmark_sigma2_smoke/`; both are gitignored):

- **`all_results.csv`** — per-`σ²` iterations, convergence flag, relative residual,
  and wall time for every solver, plus ichol/AMG/basis build times.
- **`iterations_vs_sigma2.png`** — log–log PCG iterations vs `σ²`, one curve per
  solver. Only **converged** runs (flag `== 0`) are plotted; a non-converged solve
  shows up as a **gap** in its curve.
- **`run_config.{mat,json}`** — the full parameter set and notes for the run.

---

## What the run shows

A qualitative snapshot from one `n = 3000` run (numbers vary; outputs are not
committed):

- At the hardest `σ² = 1e-8` the system is so ill-conditioned that most solvers
  fail to converge within the iteration budget — but the exact-eigenvector
  deflation `defl_P` still converges (~504 iterations).
- As `σ²` grows the systems become well-conditioned and everything converges, with
  the deflation/sketch methods dominating. Near `σ² = 1`, iteration counts land
  around: `unprec` ≈ 38, `ichol` ≈ 44, `amg` ≈ 30, `defl_P` ≈ 3, `sketch q=2` ≈ 2.

The takeaway: a recycled deflation coarse space cuts PCG iterations dramatically,
with the largest gains exactly in the ill-conditioned low-`σ²` regime — and because
the coarse space is reusable across the whole sweep, that gain comes at a one-time
construction cost.

---

## File map

**Main experiment**

| File | Role |
|------|------|
| `run_benchmark.m` | Driver — `σ²` sweep, all preconditioners, recycling. |
| `download_ccpp.m` | Fetch CCPP data → `data/ccpp.csv`. |
| `rbf_kernel_matrix.m` | Dense RBF kernel `K`. |
| `pairwise_sq_dist.m` | Squared Euclidean distances (kernel building block). |
| `estimate_median_lengthscale.m` | Median-distance lengthscale heuristic. |
| `standardize_data.m` | Zero-mean / unit-variance features and target. |
| `load_dataset_csv_or_mat.m` | Load `X`, `y` (last column = target). |
| `build_ichol_robust.m` | Incomplete-Cholesky factor with escalating diagonal shift. |

**Secondary / legacy diagnostics**

| File | Role |
|------|------|
| `run_kernel_pcg_benchmark.m` + `run_pcg_sequence.m` | Older baseline: sweeps the **lengthscale** (not `σ²`), comparing plain PCG vs ichol only; writes to `results/`. |
| `plot_pcg_results.m` | Plots for the lengthscale benchmark. |
| `plot_kernel_spectrum.m` | Eigen-spectrum overlay of `A` vs the ichol-preconditioned operator (conditioning diagnostic). |
