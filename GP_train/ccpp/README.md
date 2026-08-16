# CCPP Gaussian-process training — a sequence of dense SPD systems

Gaussian-process (GP) regression on the
[UCI Combined Cycle Power Plant dataset](https://archive.ics.uci.edu/dataset/294),
with an emphasis on **recycling preconditioners while the GP hyperparameters are
being optimized**. The main benchmark, `run_ard_training_benchmark.m`, minimizes
the negative log marginal likelihood with Adam. Every optimizer state changes
an anisotropic RBF covariance matrix and requires several linear solves with the
same matrix. The result is a sequence of dense, symmetric positive-definite
(SPD) systems whose eigenspaces generally move during training.

This directory contains two related experiments:

- **ARD training (main benchmark):** four feature lengthscales, the signal
  variance, and the noise variance are optimized. Changing the lengthscales
  changes the off-diagonal kernel entries and generally rotates the eigenspaces.
- **Noise sweep (controlled secondary benchmark):** one RBF kernel is fixed and
  only the diagonal noise/ridge term changes. Every matrix then has exactly the
  same eigenvectors, making it an invariant-subspace recycling control case.

Both experiments compare PCG with unrecycled, recycled, and freshly rebuilt
preconditioners. The central question is whether a costly spectral or
factorization-based preconditioner built for one system remains useful for the
systems that follow it.

## 1. Dataset and regression task

The Combined Cycle Power Plant (CCPP) dataset contains 9,568 hourly observations
of a power plant operated at full load. Each row has four inputs and one
regression target:

| column | quantity | role |
|---|---|---|
| `AT` | ambient temperature | feature 1 |
| `V` | exhaust vacuum | feature 2 |
| `AP` | ambient pressure | feature 3 |
| `RH` | relative humidity | feature 4 |
| `PE` | net hourly electrical energy output | target |

`download_ccpp.m` downloads `Folds5x2_pp.xlsx` from UCI and writes the numeric,
header-less file `data/ccpp.csv`, with `PE` in the last column. The `data/`
directory is gitignored because it is reproducible from the download script.

For the ARD benchmark, a seeded random permutation selects 3,000 training rows
and 1,000 disjoint test rows by default. Each feature and the target are centered
and scaled using **training-set statistics only**; the same transformation is
then applied to the test set. Thus the reported held-out RMSE is in standardized
target units. The GP uses a zero mean, which is natural after this centering.

## 2. GP model and training objective

Let $X=[x_1,\ldots,x_n]^\top\in\mathbb{R}^{n\times4}$ be the standardized
training inputs and $y\in\mathbb{R}^n$ the standardized energy outputs. The ARD
squared-exponential correlation kernel is

```math
K_{ij}(\ell)
= \exp\!\left(
  -\frac12\sum_{r=1}^{4}
  \frac{(x_{ir}-x_{jr})^2}{\ell_r^2}
  \right),
```

where one positive lengthscale $\ell_r$ is learned for each feature. The latent
function and noisy observations are modeled as

```math
f(X) \sim \mathcal N\!\left(0,\,s_f^2K(\ell)\right),
\qquad
y=f(X)+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,s_n^2I).
```

After marginalizing $f$, the observation covariance and the linear system used
throughout training are

```math
A(\theta)
=s_f^2K(\ell)+s_n^2I,
\qquad
A(\theta)\alpha=y.
```

The six optimization variables are logarithms,

```math
\theta
=\bigl(
\log\ell_1,\ldots,\log\ell_4,
\log s_f^2,\log s_n^2
\bigr)^\top,
```

so exponentiating a finite optimizer state always produces positive
hyperparameters. The code minimizes the negative log marginal likelihood (NLML)
per training observation,

```math
\mathcal L(\theta)
=\frac1n\left[
\frac12y^\top A(\theta)^{-1}y
+\frac12\log\det A(\theta)
+\frac n2\log(2\pi)
\right].
```

The data-fit term $y^\top A^{-1}y$ rewards models that explain the observations;
the log-determinant is the GP complexity penalty. This is empirical-Bayes, or
type-II maximum-likelihood, training: the covariance hyperparameters are
optimized while the latent function has already been integrated out.

The exact NLML is evaluated with a dense Cholesky factor only at initialization
and at the final state of each solver trajectory. These exact evaluations are
quality diagnostics, not the objective evaluations used to advance Adam, and
their cost is excluded from the timed setup-and-solve totals.

## 3. Gradient estimation and the required linear systems

Write $A_j=\partial A/\partial\theta_j$ and $\alpha=A^{-1}y$. Differentiating
the NLML gives

```math
\frac{\partial\mathcal L}{\partial\theta_j}
=\frac{1}{2n}
\left[
\mathrm{tr}(A^{-1}A_j)
-\alpha^\top A_j\alpha
\right].
```

The derivative matrices implemented by `ard_rbf_kernel.m` and
`ard_gp_gradient.m` are

```math
\begin{aligned}
A_r
&=s_f^2\left(K\odot\frac{D_r^2}{\ell_r^2}\right),
&&r=1,\ldots,4,\\
A_5&=s_f^2K,\\
A_6&=s_n^2I,
\end{aligned}
```

where $(D_r^2)_{ij}=(x_{ir}-x_{jr})^2$ and $\odot$ denotes entrywise
multiplication. The positive sign in the lengthscale derivative follows because
the variables are $\log\ell_r$, not $\ell_r$.

Computing $\mathrm{tr}(A^{-1}A_j)$ exactly would be too expensive. The
benchmark fixes $m=8$ independent Rademacher probe vectors by default,
$z_q\in\{-1,+1\}^n$, and uses the Hutchinson estimate

```math
\mathrm{tr}(A^{-1}A_j)
\approx
\frac1m\sum_{q=1}^{m}
z_q^\top A^{-1}A_jz_q
=\frac1m\sum_{q=1}^{m}u_q^\top A_jz_q,
\qquad
Au_q=z_q.
```

Consequently, one optimizer state requires $m+1$ systems with a common
coefficient matrix:

```math
A_k
\begin{bmatrix}
\alpha_k & u_{k,1} & \cdots & u_{k,m}
\end{bmatrix}
=
\begin{bmatrix}
y & z_1 & \cdots & z_m
\end{bmatrix}.
```

The implementation calls MATLAB `pcg` separately for the target and every probe,
with tolerance $10^{-6}$, at most 10,000 iterations, and a zero initial guess.
With the default eight probes this is **nine PCG solves per optimizer state and
270 solves per completed 30-state training trajectory**. If any solve fails, is non-finite,
or has a recomputed relative residual above $2\times10^{-6}$, that method's
trajectory stops instead of updating Adam with an unreliable gradient.

All methods use the same fixed probes. This common-random-number design removes
probe-to-probe variation from solver comparisons; it does not make the estimated
trace exact. The finite-difference test in `tests/test_ard_training_components.m`
replaces the random probes by a scaled coordinate basis, for which the trace is
exact, and verifies the implemented gradient.

## 4. Optimization: projected Adam in log-parameter space

`run_ard_training_benchmark.m` uses Adam to minimize the stochastic NLML
gradient. At completed state $k$, with gradient $g_k$, it updates

```math
\begin{aligned}
m_k&=\beta_1m_{k-1}+(1-\beta_1)g_k,\\
v_k&=\beta_2v_{k-1}+(1-\beta_2)(g_k\odot g_k),\\
\widehat m_k&=m_k/(1-\beta_1^k),\\
\widehat v_k&=v_k/(1-\beta_2^k),\\
\widetilde\theta_{k+1}
&=\theta_k-\eta\,
\widehat m_k\oslash(\sqrt{\widehat v_k}+\varepsilon_{\rm Adam}),\\
\theta_{k+1}&=\Pi_{[\theta_{\min},\theta_{\max}]}
(\widetilde\theta_{k+1}),
\end{aligned}
```

where $\odot$ and $\oslash$ are elementwise multiplication and division, and
$\Pi$ clips every log-parameter to its configured bounds.

| setting | default |
|---|---:|
| learning rate $\eta$ | 0.05 |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\varepsilon_{\rm Adam}$ | $10^{-8}$ |
| lengthscale bounds | $0.05\leq\ell_r\leq20$ |
| signal-variance bounds | $10^{-3}\leq s_f^2\leq10^3$ |
| noise-variance bounds | $10^{-6}\leq s_n^2\leq1$ |

The initial lengthscales are coordinatewise median absolute pairwise distances,
estimated from at most 1,000 training points. The initial variances are
$s_f^2=1$ and $s_n^2=0.1$, followed by projection onto the same bounds.

`NumSteps=30` means 30 **parameter states**, not 30 updates: states 1 through 29
produce Adam updates and state 30 supplies the final gradient and diagnostics.
Each solver arm starts from the same $\theta_1$, probes, Gaussian sketch, Adam
state, train/test split, and random seed, but it runs an independent optimization
trajectory. Small differences in iterative solutions can change the stochastic
gradient and therefore all later matrices; comparing methods only at a shared
step number does not imply that their $A_k$ are identical after the first state.

## 5. How the sequence of systems arises

For each solver arm the training loop is

```math
\boxed{
\theta_k
\longrightarrow A_k=A(\theta_k)
\longrightarrow
\{A_k\alpha_k=y,\ A_ku_{k,q}=z_q\}_{q=1}^{m}
\longrightarrow g_k
\longrightarrow\theta_{k+1}
}
```

This is a sequence of linear systems because the optimizer changes the
coefficient matrix after every successful gradient evaluation. The training
sample count and matrix dimension remain fixed, as do $X$, $y$, the probes,
and all coordinatewise distance matrices $D_r^2$.

What changes from $A_k$ to $A_{k+1}$ is determined by the six components of
the Adam step:

- **Lengthscales:** changing $\ell_r$ changes pairwise correlations throughout
  $K$. Different feature directions are rescaled differently, so the kernel
  eigenspaces generally rotate as well as its eigenvalues changing.
- **Signal variance:** changing $s_f^2$ scales the entire correlated part $K$,
  including its off-diagonal entries.
- **Noise variance:** changing $s_n^2$ shifts every eigenvalue by adding a
  diagonal multiple of the identity and supplies the positive spectral floor.

If the lengthscales were fixed, changing only $s_f^2$ and $s_n^2$ would preserve
the eigenvectors of $K$. ARD training is harder for recycling precisely because
the lengthscales also move.

The code records two consecutive-state drift measures:

```math
\delta_A^{(k)}
=\frac{\lVert A_k-A_{k-1}\rVert_F}{\lVert A_{k-1}\rVert_F},
\qquad
\delta_{\rm off}^{(k)}
=\frac{\lVert\mathrm{off}(s_{f,k}^2K_k)
-\mathrm{off}(s_{f,k-1}^2K_{k-1})\rVert_F}
{\lVert\mathrm{off}(s_{f,k-1}^2K_{k-1})\rVert_F}.
```

Here `off` sets the diagonal to zero. The second metric isolates changes in the
correlated, non-diagonal covariance from the diagonal noise shift. The initial
state has no predecessor, so both changes are recorded as `NaN`.

Each $A_k$ is dense and costs $O(n^2)$ storage and work per matrix-vector
product. It is symmetric, and it is positive definite because $K_k$ is positive
semidefinite and the bounded noise variance satisfies $s_{n,k}^2>0$. PCG is
therefore the appropriate Krylov method. The benchmark applies
$A_kY=s_{f,k}^2K_kY+s_{n,k}^2Y$ through a function handle. Both dense $K_k$ and
$A_k$ are also explicitly assembled and stored for preconditioner construction
and matrix-drift diagnostics.

## 6. Recycled and fresh preconditioners

The full ARD benchmark runs ten training arms. Every arm includes the time to
build or update its preconditioner and the time for all target/probe solves.

| arm | basis or factor | refresh policy |
|---|---|---|
| `unprec` | none | none |
| `exact_chol_recycle_once` | exact $A_1=L_0L_0^\top$ | build $L_0$ once |
| `defl_exact_recycle_once` | 100 largest-eigenvalue modes of $A_1$ | build $V_1$ once |
| `defl_exact_fresh_oracle` | 100 largest-eigenvalue modes of $A_k$ | rebuild at every state |
| `defl_sketch_q1_recycle_once` | $\mathrm{qr}(K_1\Omega)$ | build once |
| `defl_sketch_q1_fresh_oracle` | $\mathrm{qr}(K_k\Omega)$ | rebuild every state |
| `defl_sketch_q2_recycle_once` | $\mathrm{qr}(K_1^2\Omega)$ | build once |
| `defl_sketch_q2_fresh_oracle` | $\mathrm{qr}(K_k^2\Omega)$ | rebuild every state |
| `defl_sketch_q3_recycle_once` | $\mathrm{qr}(K_1^3\Omega)$ | build once |
| `defl_sketch_q3_fresh_oracle` | $\mathrm{qr}(K_k^3\Omega)$ | rebuild every state |

### Frozen exact Cholesky

At the first state, $L_0L_0^\top=A_1$, and the preconditioner applies

```math
M_0^{-1}r=L_0^{-\top}L_0^{-1}r=A_1^{-1}r.
```

Thus the initial preconditioned operator is the identity up to roundoff. The same
factor is frozen for every later $A_k$, where it is no longer an exact inverse.
This arm measures how quickly a high-quality factorization loses effectiveness
as the optimizer moves the kernel. There is intentionally no fresh-Cholesky arm:
a newly exact inverse at every state would make the iterative comparison
degenerate.

### Exact and sketched deflation

For an orthonormal coarse basis $V\in\mathbb R^{n\times r_c}$, all deflation
arms apply the SPD preconditioner

```math
P_k
=(I-VV^\top)
+\tau V(V^\top A_kV)^{-1}V^\top,
\qquad \tau=0.5.
```

The first term acts as the identity off the coarse space; the second replaces
the inverse action within the coarse space. When $V$ contains exact eigenvectors
of $A_k$, the corresponding eigenvalues of the symmetrically preconditioned
operator become $\tau$.

The exact bases use `eigs(...,'largestabs')`; because $A_k$ is SPD, these are
the largest eigenvalues. The sketched bases use one shared Gaussian matrix

```math
\Omega\in\mathbb R^{n\times(pr)},
\qquad p=2,\quad r=100,
```

and plain power iteration $Y=K_k^q\Omega$, followed by `qr(Y,0)`. No
Rayleigh--Ritz truncation is performed: at the full defaults, an exact basis has
100 columns while every sketch retains $pr=200$ columns. Increasing $q$
amplifies the dominant eigendirections but costs another dense kernel/block
product.

“Recycle once” freezes the basis $V_1$, not the entire preconditioner. At every
state the code recomputes the small coarse matrix $E_k=V_1^\top A_kV_1$ and its
Cholesky factor, so the coarse correction still uses the current operator.
“Fresh oracle” instead rebuilds $V_k$ as well as $E_k$. It is an intentionally
expensive reference showing the benefit available when the coarse space tracks
the changing eigenspace perfectly or approximately.

For the sketches, the same $\Omega$ is used at every state and by paired fresh
and recycled arms. For exact `eigs` bases, the random generator is reset before
each method so paired comparisons receive the same initial random start. These
controls isolate refresh policy as far as the independent optimizer trajectories
allow.

## 7. Spectrum diagnostics

After all training arms finish, `plot_ard_training_spectra.m` chooses a canonical
trajectory: `defl_exact_fresh_oracle` when it completed, otherwise the successful
method with the most recorded states. It inspects up to three distinct states:

1. the initial state;
2. the state with maximum recorded off-diagonal covariance change;
3. the final state.

At each snapshot it forms the dense matrix and computes its full spectrum. It
also builds the exact Cholesky factor $L_0$ of the canonical initial matrix once
and examines

```math
L_0^{-1}A_kL_0^{-\top}
```

without refreshing $L_0$. This is the symmetric operator relevant to the frozen
exact-factor preconditioner. The output retains the signed values of the 500
smallest-absolute and 500 largest-absolute eigenvalues (clamped for small
problems), even though materially negative values should not occur for these SPD
operators.

An idealized rank-100 ablation additionally compares no deflation against exact
deflation of the largest 100 modes, smallest 100 modes, and a 50-smallest plus
50-largest split. Selected ideal eigenvalues are replaced by $\tau=0.5$. This
diagnostic asks which spectral tail is worth targeting; it does not alter the
training arms.

Full dense eigendecompositions, exact final NLML, prediction, and held-out RMSE
are deliberately excluded from preconditioner setup and PCG solve timings.

## 8. Running the ARD benchmark

From `GP_train/ccpp/` in MATLAB:

```matlab
download_ccpp                 % one-time download; no-op if data already exists
out = run_ard_training_benchmark;
```

The default experiment is intentionally expensive: ten independent trajectories
each use 3,000-by-3,000 dense kernels, 30 parameter states, nine PCG right-hand
sides per state, and fresh-oracle eigenspace builds where applicable.

For a fast end-to-end run:

```matlab
out = run_ard_training_benchmark('SmokeTest', true);
```

Smoke mode caps the configuration at 300 training points, 100 test points, three
states, two probes, rank 20, and 500 PCG iterations. Other parameters can be
overridden with name-value arguments, for example:

```matlab
out = run_ard_training_benchmark( ...
    'NTrain', 1500, ...
    'NTest', 500, ...
    'NumSteps', 20, ...
    'NumProbes', 6, ...
    'Rank', 60, ...
    'SolverKeys', {'unprec', 'defl_exact_recycle_once', ...
                   'defl_exact_fresh_oracle'});
```

| option | full default | meaning |
|---|---:|---|
| `NTrain` | 3000 | training observations and system dimension |
| `NTest` | 1000 | held-out observations |
| `NumSteps` | 30 | optimizer parameter states |
| `NumProbes` | 8 | fixed Hutchinson probes per state |
| `Tol` | $10^{-6}$ | PCG relative-residual tolerance |
| `MaxIt` | 10000 | PCG iteration limit |
| `Rank` | 100 | exact-deflation rank and sketch base rank |
| `Tau` | 0.5 | deflated coarse eigenvalue |
| `SketchQList` | `[1 2 3]` | numbers of kernel power applications |
| `SketchOversample` | 2 | sketch width multiplier |
| `AdamRate` | 0.05 | Adam learning rate |
| `SpectrumCount` | 500 | retained values per absolute spectral tail |
| `Seed` | 0 | data, probe, sketch, and eigensolver reproducibility seed |
| `SolverKeys` | all methods | optional subset of registry keys |

`DataFile` accepts a numeric CSV or a MAT-file containing exactly one numeric
matrix; the last column is always treated as the target. The ARD driver requires
exactly four feature columns. `OutDir` overrides the output directory.

### ARD outputs

Results are written to `benchmark_ard_training/`, or
`benchmark_ard_training_smoke/` in smoke mode:

| output | contents |
|---|---|
| `solve_results.csv` | one row per method, state, and target/probe solve: iterations, flags, residuals, and solve time |
| `training_results.csv` | hyperparameters, gradient, matrix drift, setup components, and completion status per state |
| `summary.csv` | completed states, convergence fraction, total work/time, exact final NLML, RMSE, and final hyperparameters |
| `run_config.mat`, `run_config.json` | parameters, method registry, data split, initialization, and standardization statistics |
| `benchmark_results.mat` | consolidated MATLAB result structure, including spectra |
| `spectrum_values.csv` | signed and absolute values from both spectral tails |
| `spectrum_summary.csv` | condition estimates, tail gaps, noise-floor distances, and subspace overlaps |
| `spectrum_full.mat` | complete eigenvalue vectors and snapshot metadata |
| `iterations_per_step.png` | total target-plus-probe PCG iterations per state |
| `hyperparameter_trajectories.png` | canonical six-parameter training path |
| `matrix_change.png` | canonical off-diagonal covariance drift |
| `quality_vs_runtime.png` | exact final NLML versus timed setup-plus-solve cost |
| `spectrum_raw_and_recycled_exact.pdf` | raw and frozen-exact-factor spectral tails |
| `spectrum_tail_ablation.pdf` | ideal largest/smallest/two-tail deflation comparison |

The output directories are gitignored and can be regenerated.

## 9. Secondary experiment: fixed-kernel noise sweep

`run_benchmark.m` isolates the effect of changing only the observation-noise or
ridge parameter. It standardizes all data, selects 3,000 rows, chooses one
isotropic RBF lengthscale by the median pairwise-distance heuristic, and builds
one kernel $K$. It then solves

```math
A(\sigma_j^2)x_j=y,
\qquad
A(\sigma_j^2)=K+\sigma_j^2I,
\qquad
\sigma_j^2\in\mathrm{logspace}(-8,0,10).
```

This is a prescribed parameter sweep, not an optimization: $x_j$ does not
determine $\sigma_{j+1}^2$, and there are no Hutchinson probes or likelihood
gradients. Its sequence has the exact eigendecomposition

```math
K=Q\Lambda Q^\top
\quad\Longrightarrow\quad
A(\sigma_j^2)=Q(\Lambda+\sigma_j^2I)Q^\top.
```

Therefore all systems share the same eigenvectors. A basis computed from $K$
once is exactly the same basis that would be obtained from $K$ later in the
sweep (although a sketched basis is still only an approximation to the dominant
eigenspace). This is the clean invariant-subspace case against which the harder
ARD sequence should be understood.

The sweep compares unpreconditioned PCG, robust incomplete Cholesky, a two-level
smoothed-aggregation AMG V-cycle, exact dominant-mode deflation, $q=1,2,3$
sketched deflation, and an incomplete-Cholesky-split two-level method. The
K-based exact and sketch bases are recycled by default; incomplete Cholesky and
AMG are rebuilt at every noise value. The split two-level basis is only
approximately reusable because its operator
$\widehat A_j=L_j^{-1}A_jL_j^{-\top}$ changes with the rebuilt incomplete
Cholesky factor $L_j$.

Run it with:

```matlab
run_benchmark
```

For its smoke configuration, set the base-workspace toggle before running the
script:

```matlab
SMOKE_TEST = 1;
run_benchmark
```

The full sweep writes `all_results.csv`, `iterations_vs_sigma2.png`, and
`run_config.{mat,json}` to `benchmark_sigma2/`; smoke mode writes to
`benchmark_sigma2_smoke/`. Non-converged PCG runs remain in the CSV but appear
as gaps in the iteration plot. No fixed numerical performance claims are made
here because benchmark results are not committed and depend on the MATLAB and
hardware environment.

## 10. File map

| file | role |
|---|---|
| `run_ard_training_benchmark.m` | ARD training driver, method registry, Adam loop, timing, tables, and training plots |
| `ard_rbf_kernel.m` | ARD kernel, log-lengthscale derivatives, and reusable coordinate-distance matrices |
| `ard_gp_gradient.m` | Hutchinson NLML-gradient estimator |
| `ard_rbf_cross_kernel.m` | train-to-test covariance for held-out prediction |
| `plot_ard_training_spectra.m` | representative-state spectra and tail-ablation outputs |
| `ideal_deflated_spectrum.m` | ideal exact-tail deflation spectrum |
| `extract_spectrum_tails.m` | absolute-tail ranking while retaining eigenvalue signs |
| `run_benchmark.m` | fixed-kernel noise/ridge sweep |
| `rbf_kernel_matrix.m` | isotropic dense RBF kernel used by the sweep |
| `pairwise_sq_dist.m` | squared-distance building block for the isotropic kernel |
| `estimate_median_lengthscale.m` | isotropic median-distance initialization |
| `build_ichol_robust.m` | incomplete Cholesky with an escalating diagonal shift |
| `download_ccpp.m` | reproducible UCI download and CSV conversion |
| `load_dataset_csv_or_mat.m` | numeric CSV/MAT loader using the last column as target |
| `standardize_data.m` | whole-dataset standardization used by the secondary sweep |
| `tests/test_ard_training_components.m` | kernel, Cholesky, gradient, spectrum-tail, and cross-kernel checks |

The deflation and AMG implementations are shared repository utilities under
[`+src/+precond/`](../../+src/+precond). Both benchmark drivers add the repository
root to the MATLAB path automatically.

`run_kernel_pcg_benchmark.m` with `run_pcg_sequence.m` is an older diagnostic
that sweeps the RBF lengthscale and compares plain PCG with incomplete Cholesky;
`plot_pcg_results.m` and `plot_kernel_spectrum.m` support that legacy workflow.

## 11. References

- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
  Learning*, Chapter 2: [Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
- K. M. Cutajar, M. A. Osborne, J. P. Cunningham, and M. Filippone,
  [Preconditioning Kernel Matrices](https://proceedings.mlr.press/v48/cutajar16.html),
  ICML 2016.
- J. Wenger, G. Pleiss, P. Hennig, J. Cunningham, and J. Gardner,
  [Preconditioning for Scalable Gaussian Process Hyperparameter
  Optimization](https://proceedings.mlr.press/v162/wenger22a.html),
  ICML 2022.
- [Combined Cycle Power Plant](https://archive.ics.uci.edu/dataset/294),
  UCI Machine Learning Repository, dataset 294.
