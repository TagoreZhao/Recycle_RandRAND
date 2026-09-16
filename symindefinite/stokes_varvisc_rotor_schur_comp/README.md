# Schur complement of variable-viscosity immersed-rotor Stokes

## Two-stage shared-subspace augmentation in the main benchmark

The default main benchmark includes `deflate_sequential_shared_subspace_augmented`
alongside the original two-stage shared-subspace solver. This is the only
main-benchmark arm with fresh Arnoldi augmentation; the other deflation
methods retain their existing algorithms.

```matlab
addpath('symindefinite/stokes_varvisc_rotor_schur_comp');
maxNumCompThreads(4);
SMOKE_TEST = true; run_varvisc_schur_recycle;
SMOKE_TEST = false; run_varvisc_schur_recycle;
% Regenerate plots from saved CSVs:
replot_varvisc_schur;
```

Both two-stage arms share the complete cached basis
`V = orth([V_large,V_small])`, its refresh schedule, and both tau values.
With current defaults this contains 200 Gaussian large-mode columns and
20 Lanczos small-mode columns, subject to numerical rank. At each step the
augmented arm builds `W` using two-pass Arnoldi on `Pi*S_i*Pi`, starting
from `Pi*(rhs_i-S_i*x0)`, where `Pi=I-V*V'` and `x0` is the shared warm start
from the previous reference solution. Thus augmentation uses the current
residual, including the nonzero initial guess.

The **same `Z=[V,W]` is used in both stages**. If
`D(Z,A,tau)=I-Z*Z'+tau*Z*(Z'*A*Z)^(-1)*Z'`, the construction is
`H1=D(Z,S_i,tau1)^(1/2)`, `S1=H1*S_i*H1`,
`P2=D(Z,S1,tau2)`, and the PCG inverse preconditioner is `H1*P2*H1`.
Both coarse matrices are built from their current operators. `W` is held
fixed during PCG and discarded afterward; it never enters the recycled cache.
`AUGMENT_M=20` is set in `make_varvisc_schur_params`; zero gives the exact
unaugmented two-stage path. Breakdown is recorded without random padding.

The four-case full run uses 60 timesteps, `h=0.05`, `dt=0.02`, and seed 1.
Main-benchmark PCG and reference settings are shared across all arms.
Outputs under `varvisc_schur_recycle/` include the new arm in
`all_results.csv`, all-solver plots and cross-case summaries, plus
`two_stage_augmentation_summary.csv`, `two_stage_augmentation_report.md`,
and each case's linear `two_stage_augmentation_comparison.png`.
The master CSV records true and reported residuals, reference-solution errors,
base/added dimensions, Arnoldi status, orthogonality, recurrence error,
Arnoldi time, and Arnoldi/residual operator columns. `chol_flag` records the
frozen-Cholesky solver flag; `factor_chol_flag` separately records the Schur
factorization diagnostic. The dedicated summary compares
iterations with the unaugmented two-stage solver and retains failed solves.
Iteration reductions alone do not establish a runtime improvement.

The completed full run measured:

| Case | Two-stage iterations | Augmented two-stage iterations | Iterations saved |
|---|---:|---:|---:|
| `bar_rotating_nu_orbiting` | 7,777 | 6,617 | 14.92% |
| `disk_translating_nu_wake` | 7,107 | 5,065 | 28.73% |
| `disk_static_nu_checkerboard_shift` | 8,766 | 6,260 | 28.59% |
| `disk_static_nu_const` | 127 | 110 | 13.39% |

All 1,920 solves across eight arms and four 60-step cases met the accuracy
criteria. Every augmented solve used 220 recycled columns plus 20 fresh
directions; the largest augmented true residual was `9.9998e-9` and the
largest relative reference-solution error was `3.175e-8`. In the constant
control, both two-stage arms need zero PCG iterations from step 19 onward
because their shared warm start already meets tolerance. The 14-test suite
and all 24 native smoke solves also passed.

See the [full two-stage report](varvisc_schur_recycle/two_stage_augmentation_report.md)
for the paired summary and iteration-versus-step plots for all four cases.

## Standalone inverse-Gaussian augmentation

```matlab
addpath('symindefinite/stokes_varvisc_rotor_schur_comp');
addpath('symindefinite/stokes_varvisc_rotor_schur_comp/tests');
test_varvisc_schur_augmentation;
test_varvisc_schur_augmented_workflow;
run_varvisc_schur_augmented_benchmark('smoke');
run_varvisc_schur_augmented_benchmark('full');
% Resume a seed/case, or regenerate the completed full report:
run_varvisc_schur_augmented_benchmark('full',2,1);
run_varvisc_schur_augmented_benchmark('finalize');
```

This two-arm experiment transfers the parent's
[projected-Arnoldi augmentation](../stokes_varvisc_rotor/GAUSSIAN_AUGMENTED_SOLVE.md)
to the reduced SPD Schur system. Both arms share the frozen basis
`V = orth(S_1^(-2)*Omega)`, computed using two exact inverse applications at
step 1. All numerically independent columns are retained in the fixed reduced
pressure/multiplier coordinates. At every timestep the augmented arm builds
up to 20 fresh directions from
`K_20((I-V*V')*S_i*(I-V*V'), (I-V*V')*rhs_i)` using two-pass projected
Arnoldi. It discards those directions after solving; only `V` is recycled.
Breakdown and limited complement dimensions are recorded without random padding.

With `Z=V` or `Z=[V,W_i]`, the inverse preconditioner is
`P = I-Z*Z' + tau*Z*(Z'*S_i*Z)^(-1)*Z'`. This uses the existing Schur
coarse correction and PCG, with a common frozen `tau=lambda_max(S_1)`.
Both coarse matrices use the current operator. PCG starts from zero, uses
tolerance `1e-8`, and has a 4,000-iteration cap. The dense Schur matrix and
its exact Cholesky are needed only initially; later augmentation, coarse
setup, and PCG use operator handles. The velocity factor within the current
Schur operator is still rebuilt every timestep.

Full runs use `h=0.05`, 60 solves, `dt=0.02`, seeds 1–3, and 1,000 Gaussian
columns (nominal rank 500 with oversampling 2). Cases are the rotating bar
with orbiting viscosity, the translating disk with a viscosity wake, and the
static constant-viscosity disk. Smoke runs use all cases/seeds at `h=0.16`,
16 Gaussian columns and three solves, retaining `m=20` and the full motion
period. A saved mesh and direct-KKT reference state advance keep physical
systems identical across seeds and solver arms. The driver sets four MATLAB
compute threads and records the MATLAB version and thread count.

Results are written to `varvisc_schur_augmented/` and its `_smoke` sibling:
`augmentation_results.csv`, `augmentation_summary.csv`,
`augmentation_paired_summary.csv`, `augmentation_validation.csv`, and
`augmentation_report.md`. Plots show iterations, cumulative attributed time
and forward operator work, true residuals, solution error, residual histories,
and Arnoldi orthogonality. Complete per-step PCG histories and small Arnoldi
matrices are retained under each case's `diagnostics/` directory.

Timing includes full attributed initial setup for each arm, current Arnoldi,
coarse construction, PCG and velocity recovery. Reference solves, diagnostics
and output are excluded. Forward Schur operator columns include Arnoldi and
coarse setup as well as PCG and initial tau iteration; inverse RHS columns
and dense-materialization velocity RHS columns are separate. Thus lower PCG
iteration counts need not mean less total work or time. The initial exact
factor also permits a direct solve, so this compares recycling strategies.

Validation requires flag zero, true Schur and recovered KKT residuals at most
`1e-8`, and relative solution error at most `1e-5`. Failed solves remain in
the output and aggregate totals; paired summaries also give iteration savings
on pairs meeting all accuracy criteria. No speedup is assumed. Coordinate-map
changes stop the experiment, and resumption rejects incompatible configuration.
An optional fourth argument selects an isolated output directory; use it with
`('finalize',[],[],output_root)` to regenerate a saved smoke report.

The completed full experiment measured these medians of paired seed results:

| Case | PCG iterations saved | Augmented/baseline forward work | Augmented/baseline time |
|---|---:|---:|---:|
| `bar_rotating_nu_orbiting` | 46.30% | 0.932x | 0.651x |
| `disk_translating_nu_wake` | 35.79% | 0.957x | 0.728x |
| `disk_static_nu_const` | 29.06% | 1.019x | 0.960x |

Augmentation reduced both operator work and attributed time in the moving
cases. The static control saved PCG iterations but increased total forward
operator columns; its time ratio ranged from 0.918x to 1.031x across seeds.
All 1,080 solves met the accuracy criteria, with maximum Schur residual
`9.998e-9`, KKT residual `8.414e-9`, and relative solution error `4.831e-8`.
Every augmented solve retained 1,000 base columns and added all 20 directions.
The [full report](varvisc_schur_augmented/augmentation_report.md) contains seed
ranges, cost components, validation data, and figures. The 13-test regression
suite and all 54 smoke solves also passed.

## Paired Gaussian recycling versus rebuilding

```matlab
addpath('symindefinite/stokes_varvisc_rotor_schur_comp');
addpath('symindefinite/stokes_varvisc_rotor_schur_comp/tests');
maxNumCompThreads(4);
test_varvisc_schur_gaussian_refresh;
run_varvisc_schur_gaussian_refresh('smoke');
run_varvisc_schur_gaussian_refresh('full');
% Resume one seed/case or replot the saved full experiment:
run_varvisc_schur_gaussian_refresh('full',2,1);
run_varvisc_schur_gaussian_refresh('finalize');
```

This focused experiment compares `orth(S_1^(-2)*Omega)`, built once and
recycled, against `orth(S_i^(-1)*Omega)`, rebuilt every timestep. The power
count keeps its existing meaning: `q=2` applies the inverse twice and `q=1`
once. Each rebuilt sketch uses an exact Cholesky factorization of the current
reduced Schur matrix, applied through triangular solves. All returned columns
of the final `orth` are retained, with any numerical rank loss reported.
The recycled basis stays in the fixed reduced pressure/multiplier coordinates.

Both arms use zero-start PCG at tolerance `1e-8`, a 4,000-iteration cap, and
the existing SPD correction `I-V*V' + tau*V*(V'*S_i*V)^(-1)*V'`. Their coarse
matrices use the current Schur operator. A deterministic eigenvalue solve sets
one common `tau=lambda_max(S_1)` per case, frozen for the entire sequence.

The full experiment uses `h=0.05`, 60 steps, `dt=0.02`, seeds 1–3, and 1,000
Gaussian columns (nominal rank 500, oversampling 2). It includes
`bar_rotating_nu_orbiting`, `disk_translating_nu_wake`, and
`disk_static_nu_const`. A saved mesh and direct-KKT reference trajectory keep
the physical systems identical across seeds. Within each seed, one Gaussian
start block is shared across both arms and all timesteps. The smoke run uses
all three cases/seeds with `h=0.16`, 16 columns and three steps, preserving
the full motion period.

Results are saved under `varvisc_schur_gaussian_refresh/` (or the `_smoke`
sibling): `comparison_results.csv`, `comparison_summary.csv`,
`comparison_report.md`, convergence/work figures and complete residual
histories in per-case `diagnostics/`. Per-case checkpoints are configuration
checked; resume/finalize with the same MATLAB version and thread count.
Changing reduced dimensions or coordinate masks stops the experiment rather
than silently reusing an incompatible basis.

Timing includes attributed Schur construction, dense materialization, exact
Cholesky, initial tau selection, basis construction, coarse setup, PCG and
velocity recovery. Reference solves, diagnostics and output are excluded;
shared factors are charged fully to each algorithm requiring them. Exact
Cholesky also permits a direct solve, so these timings compare the two basis
strategies without claiming an advantage over direct solution. Inverse RHS,
materialization velocity RHS and forward Schur operator counts are separate.

Two inverse applications provide stronger small-eigenvalue filtering; one
current inverse application still produces an approximate subspace. This
experiment changes both power count and refresh cadence, so the result cannot
isolate the effect of refreshing at fixed `q`. First-step and constant-control
results expose initial basis quality without operator aging. PCG-reported
histories, recomputed Schur residuals, recovered KKT residuals and reference
solution errors are retained separately, including tolerance failures.

The completed full run gave the following median total PCG iterations per
60-step trajectory across the three seeds:

| Case | Recycled `q=2` | Rebuilt `q=1` | Rebuilt/recycled attributed time |
|---|---:|---:|---:|
| `bar_rotating_nu_orbiting` | 16,440 | 14,960 | 1.44x |
| `disk_translating_nu_wake` | 15,824 | 11,689 | 1.44x |
| `disk_static_nu_const` | 4,147 | 5,095 | 2.76x |

Refreshing reduces iterations on the moving Schur systems, but its repeated
setup cost outweighs those savings in this run. All 1,080 solves returned
zero PCG flags and met `1e-8` in both the recomputed Schur and recovered KKT
residuals. Every basis retained 1,000 columns; the largest inverse-probe
residual was `6.82e-14`. The [full report](varvisc_schur_gaussian_refresh/comparison_report.md)
includes seed ranges, accuracy results, cost components and convergence plots.

This benchmark is the reduced-system counterpart of
[`../stokes_varvisc_rotor/`](../stokes_varvisc_rotor/). The parent problem
produces a sequence of sparse, symmetric indefinite Stokes KKT systems. This
benchmark eliminates velocity and applies the Schur complement in the pressure
and immersed-multiplier variables through a function handle.

The distinction matters because the reduced operator has different algebraic
properties from the KKT matrix: after removal of the pressure-pin direction it
is symmetric positive definite (SPD), but its matrix representation is dense.
Variable viscosity also changes the velocity inverse inside the Schur
complement, so its current apply must be reconstructed at every time step. The border-only shortcut
available to the constant-viscosity immersed-rotor problem does not apply.

## 1. The variable-viscosity KKT system

The fixed fluid mesh has `N` P1 nodes. At time step $n$, the unknown ordering is

```math
x^n=[\,u^n;\,p^n;\,\lambda^n\,],
```

with $n_U=2N$ velocity unknowns, $n_P=N$ pressure unknowns, and $n_C(t_n)$
immersed constraints. Pressure enforces incompressibility, while $\lambda$
enforces the prescribed rigid velocity at Lagrange points carried by the solid.

Before boundary elimination, the backward-Euler system is

```math
\mathcal K_n=
\begin{bmatrix}
A_{\mathrm{vel},n}&B^\mathsf{T}&C_n^\mathsf{T}\\
B&-L_{p,\varepsilon}^n&0\\
C_n&0&0
\end{bmatrix},
\qquad
\mathcal K_n
\begin{bmatrix}u^n\\p^n\\\lambda^n\end{bmatrix}
=
\begin{bmatrix}b_{u,n}\\0\\g_n\end{bmatrix},
```

where

```math
A_{\mathrm{vel},n}
=\frac{M_2}{\Delta t}+A_2(\nu^n),
\qquad
A_2(\nu^n)=\mathrm{blkdiag}(K_\nu^n,K_\nu^n),
```

```math
(K_\nu^n)_{ij}
=\sum_{e\in\mathcal T_h}\nu_e(t_n)
\int_e\nabla\phi_i\!\cdot\!\nabla\phi_j,
```

and the elementwise Brezzi-Pitkäranta pressure stabilization is

```math
(L_{p,\varepsilon}^n)_{ij}
=\sum_{e\in\mathcal T_h}\varepsilon_e^n
\int_e\nabla\phi_i\!\cdot\!\nabla\phi_j,
\qquad
\varepsilon_e^n=\frac{h_0^2}{12\nu_e(t_n)}.
```

The coupling $C_n=C(t_n)$ evaluates velocity at the current immersed Lagrange
points, and $g_n$ contains their prescribed rigid velocities. The velocity
right-hand side carries the previous state,
$b_{u,n}=M_2u^{n-1}/\Delta t$, plus an optional body-force term.

Velocity Dirichlet values and one pressure value are imposed directly on the
sparse blocks: known columns are lifted to the block right-hand sides, the
corresponding rows and columns are zeroed, and unit constrained diagonals are
inserted. This is algebraically identical to `apply_dirichlet_sym`, but avoids
assembling the enclosing KKT matrix during ordinary Schur solves.

## 2. Constructing the Schur complement

After symmetric boundary elimination and pressure pinning, partition the KKT
system between velocity and the combined constraint variable
$y=[\,p;\lambda\,]$:

```math
\begin{bmatrix}
A_n&G_n^\mathsf{T}\\
G_n&-D_n
\end{bmatrix}
\begin{bmatrix}u\\y\end{bmatrix}
=
\begin{bmatrix}b_1\\b_2\end{bmatrix}.
```

Here $A_n\in\mathbb R^{n_U\times n_U}$ is the boundary-eliminated velocity
block, $G_n\in\mathbb R^{(n_P+n_C)\times n_U}$ contains the surviving rows of
$B$ and $C_n$, and $D_n$ is defined by the sign convention that the lower-right
KKT block is $-D_n$. Away from the pressure pin,

```math
G_n=\begin{bmatrix}\widehat B\\\widehat C_n\end{bmatrix},
\qquad
D_n=\begin{bmatrix}
L_{p,\varepsilon}^n&0\\
0&0
\end{bmatrix}.
```

Hats indicate that columns belonging to prescribed velocity degrees of freedom
have been zeroed by symmetric elimination.

The first block row gives

```math
A_nu+G_n^\mathsf{T}y=b_1
\quad\Longrightarrow\quad
u=A_n^{-1}(b_1-G_n^\mathsf{T}y).
```

Substituting this expression into $G_nu-D_ny=b_2$ yields

```math
G_nA_n^{-1}(b_1-G_n^\mathsf{T}y)-D_ny=b_2,
```

and hence the positive-sign Schur system

```math
\boxed{
S_n=D_n+G_nA_n^{-1}G_n^\mathsf{T}},
\qquad
\boxed{
S_ny=G_nA_n^{-1}b_1-b_2}.
```

After solving for $y$, velocity is recovered with

```math
u=A_n^{-1}(b_1-G_n^\mathsf{T}y).
```

### Construction in the code

`varvisc_schur_step_operator` implements the formulas directly:

```matlab
dA = decomposition(A, 'chol');
Sapply = @(X) Dred*X + Gred*(dA\(Gtred*X));
rhs_S = Gred*(dA\b1) - b2red;
```

Both sparse orientations `Gred` and `Gtred` are stored when the step is built.
The operator handle therefore performs only sparse matrix products and the
required velocity-block solve; it never transposes a large matrix while PCG or
a randomized basis builder is iterating.

`st.apply(X)` supports both vectors and block matrices, so PCG, Gaussian
sketches, Lanczos, and coarse Galerkin products all use the same matrix-free
operator. `st.to_dense()` is the explicit escape hatch for exact Cholesky,
exact spectra, rank diagnostics, and exported dense examples. Materialization
performs the block solve with every retained column of $G_n^\mathsf{T}$ and
averages the result with its transpose to remove roundoff-level asymmetry.

The returned reduced dimension is

```math
n_S=n_P+n_C-1,
```

because the pinned pressure index is removed as described next.

### Pressure-pin removal and recovery

Symmetric pressure pinning sets the corresponding KKT diagonal to $+1$ and
decouples its row and column. Since the lower-right KKT block is written as
$-D_n$, this produces

```math
(D_n)_{\mathrm{pin},\mathrm{pin}}=-1,
\qquad
(G_n)_{\mathrm{pin},:}=0.
```

The same coordinate is consequently a completely decoupled $-1$ direction in
the unreduced `Sfull`. It is the sole negative Schur direction. The code deletes
that row and column before calling PCG, solves the remaining SPD system, then
scatters the prescribed pressure value back into $y$ before recovering $u$.
Nothing is approximated by this deletion: solving the reduced system and
recovering the full vector agrees with `K\b` to roundoff.

## 3. Properties of the reduced operator

### Symmetric and positive definite after pin removal

$A_n$ is SPD after velocity boundary elimination, so
$G_nA_n^{-1}G_n^\mathsf{T}$ is symmetric positive semidefinite. Positive
element viscosity makes the pinned pressure-stabilization block positive
definite on the retained pressure coordinates. Together with the independent
immersed constraints, this makes the retained $S_n$ SPD in the benchmark.

This is different from the original KKT matrix, whose negative pressure block
and zero multiplier block make it indefinite. MINRES is appropriate for that
symmetric indefinite system; PCG is appropriate for the reduced SPD Schur
system. Applying PCG to `Sfull` before deleting the negative pin direction would
violate PCG's assumptions.

### Dense even though the KKT matrix is sparse

The assembled KKT blocks are sparse, but $A_n^{-1}$ is generally dense. The
product $G_nA_n^{-1}G_n^\mathsf{T}$ therefore couples nearly every retained
pressure and multiplier coordinate. The mathematical matrix is dense even
though ordinary applications do not form it. An incomplete sparse
factorization is not a natural preconditioner for this operator; the
exact-factor baseline explicitly materializes $S_1$ and uses dense Cholesky.

### Exact reduction of the KKT system

The Schur solve changes the algebraic representation, not the discretization.
It uses the same viscosity, stabilization, immersed coupling, forcing, boundary
values, and pressure pin as the parent KKT solve. In particular:

- the KKT system and Schur system have the same retained solution;
- prescribed velocity values are preserved during recovery;
- no approximation to $A_n^{-1}$ is used while constructing $S_n$; and
- solver tolerances affect only the iterative Schur solve, not the definition
  of the reduced operator.

## 4. How the Schur complement changes in time

Expanding the retained pressure and multiplier blocks makes the dependencies
visible:

```math
S_n=
\begin{bmatrix}
L_{p,\varepsilon}^n
 +\widehat B A_n^{-1}\widehat B^\mathsf{T}
&\widehat B A_n^{-1}\widehat C_n^\mathsf{T}\\
\widehat C_n A_n^{-1}\widehat B^\mathsf{T}
&\widehat C_n A_n^{-1}\widehat C_n^\mathsf{T}
\end{bmatrix},
```

with the pinned pressure row and column omitted. This expansion shows that
moving viscosity affects more than the explicit pressure-stabilization term:
because it changes $A_n^{-1}$, it propagates into every block of $S_n$.

### What stays constant

The following data are built once and reused:

- the fixed mesh, element geometry, and `triangulation` search object;
- the velocity mass contribution $M_2/\Delta t$;
- the uneliminated divergence matrix $B$;
- the velocity boundary degree-of-freedom set and pressure-pin location; and
- the unit-stiffness assembly data used to form coefficient-weighted matrices.

For the production cases, the retained pressure dimension and immersed
constraint count also remain fixed. In the general coupling engine, Lagrange
points outside the channel may be dropped, in which case $n_C$ and the Schur
dimension can change.

### What changes at step n

- **Element viscosity.** The moving coefficient changes the samples
  $\nu_e(t_n)$.
- **Velocity block and inverse.** $A_2(\nu^n)$ changes throughout its existing
  sparsity pattern, so $A_n$, its Cholesky factor, and its inverse action all
  have to be refreshed.
- **Pressure stabilization.** Since
  $\varepsilon_e^n=h_0^2/(12\nu_e^n)$, $L_{p,\varepsilon}^n$ changes in the
  inverse direction to viscosity.
- **Immersed coupling.** Solid motion changes the host elements and barycentric
  weights in $C_n$, as well as the constraint values $g_n$.
- **Right-hand side.** It carries $u^{n-1}$ forward and includes the current
  rigid velocity, optional force, and prescribed boundary data.
- **Complete Schur operator.** The pressure-pressure, pressure-multiplier, and
  multiplier-multiplier blocks can all change. The function-handle closure and
  reduced right-hand side are rebuilt without forming the full dense matrix.

### Contrast with the constant-viscosity Schur benchmark

In `stokes_immersed_rotor_schur_comp`, $A$, $A^{-1}$, and the
pressure-pressure block are constant. Only $C_n$ moves, so the current Schur
matrix can be assembled after only $n_C$ new velocity backsolves, and the
update satisfies the multiplier-border bound

```math
\mathrm{rank}(S_n-S_m)\le 2n_C.
```

That shortcut is invalid here. Variable viscosity changes both $A_n$ and
$L_{p,\varepsilon}^n$, so each step requires all $n_P+n_C$ velocity backsolves,
the pressure-pressure block moves, and the old $2n_C$ rank bound can be
exceeded. `run_varvisc_schur_rank` and `test_varvisc_schur_structure` verify
these differences.

The `disk_static_nu_const` case is the negative control. Its solid is stationary
and $\nu\equiv1$, so $A_n$, $D_n$, $C_n$, and $S_n$ remain constant. A frozen
step-1 inverse is therefore exact for the whole sequence.

The adversarial `disk_static_nu_checkerboard_shift` case isolates the opposite
regime. Its solid and $C_n$ are stationary, while a smooth 100:1 log-viscosity
checkerboard translates by half a wavelength. At the final step, every
high-viscosity region occupies the initial low-viscosity region and vice versa.
This complementary motion spreads the generalized eigenvalues of
$(S_n,S_1)$ toward both tails. That is what makes `chol(S_1)` stale: a
full-rank update or a large Frobenius change alone is insufficient, and a
uniform rescaling could still be easy for PCG.

## 5. Solver arms and reuse lifecycles

All arms solve the same scaled Schur right-hand side and use the same warm
start.

| key | method |
|---|---|
| `pcg_unprec` | unpreconditioned PCG |
| `chol` | exact dense `chol(S_1)`, frozen for the sequence |
| `deflate_shared_small` | direct deflation with the centrally shared smallest-mode basis |
| `deflate_gaussian_large` | Gaussian forward-power sketch for the largest modes |
| `deflate_sequential_shared_subspace` | two-stage deflation; the same concatenated small+large basis is used in both stages |
| `deflate_sequential_shared_subspace_augmented` | the same two-stage construction using the full shared basis plus fresh projected-Arnoldi directions in both stages |
| `deflate_concatenated_once` | one standard deflator built from the concatenated small+large basis |
| `deflate_adaptive_small_lift_large` | adaptive small-mode lift followed by large-mode deflation of the lifted operator |

The default nominal dimensions are `sm_eig=20` and `lg_eig=100`. Lanczos
returns `sm_eig` small vectors. Every Gaussian construction draws
`ceil(sketch_oversampling*k)` columns and retains the entire orthogonalized
basis. With the default `sketch_oversampling=2`, the large-tail bases contain
`2*lg_eig` vectors, the adaptive post-lift large basis contains
`2*(sm_eig+lg_eig)` vectors, and the inverse-Gaussian small basis contains
`2*sm_eig` vectors. The standalone, sequential, and concatenated arms reuse
one large-tail random draw and basis built from the original $S_n$. The
adaptive arm keeps an independent draw because it sketches the lifted operator.

The central small source is selected by `small_basis_source`:

- `lanczos` (default) runs fully reorthogonalized Lanczos directly on the
  current Schur apply. It computes `sm_eig+1` Ritz pairs and retains the
  first `sm_eig`; no Cholesky factorization of the Schur matrix is used by the
  Lanczos iteration itself.
- `inverse_gaussian` applies the exact current Cholesky inverse to a Gaussian
  block, performs no intermediate reorthogonalization, orthogonalizes once at
  the end, and retains every oversampled sketch vector.

All Gaussian sketches use

```math
m_{\rm sketch}=\min\!\left(n,
\left\lceil\texttt{sketch\_oversampling}\,k\right\rceil\right).
```

The basis is `orth(Y)` with no projected eigendecomposition, Rayleigh--Ritz
rotation, selection, or truncation. Standard sketches do not reorthogonalize
between subspace-iteration products. The transformed post-lift sketch
reorthogonalizes after every product because the strong lift can otherwise
collapse the enlarged block numerically. Standard large sketches use nominal
rank `k=lg_eig` and power count `q`; the post-lift sketch uses nominal rank
`k=sm_eig+lg_eig` and power count `lift_large_q`.

The reusable objects have distinct refresh rules:

1. The Cholesky of $A_n$ used by the current Schur apply is rebuilt every step
   because viscosity changes $A_n$.
2. The `chol` solver arm deliberately freezes the dense Cholesky of $S_1$ and
   reuses it as a preconditioner for later $S_n$. This is the factor that
   becomes stale in moving-viscosity cases.
3. `SMALL_BASIS_REFRESH` controls the one shared small basis.
4. `DEFLAT_SHARED_LARGE_REFRESH` controls the original-$S_n$ large basis
   shared by the standalone, sequential, and concatenated arms.
   `DEFLAT_ADAPTIVE_LIFT_LARGE_REFRESH` independently controls the transformed
   post-lift basis. Step 1 always builds each enabled object; a finite value $R$
   rebuilds it at steps $1,1+R,1+2R,\ldots$. Every interval defaults to `Inf`.

A shared-small refresh causes the sequential and one-shot designs to recombine
the shared large basis with the new small basis. It also forces the adaptive
post-lift large sketch to rebuild because that sketch acts on a newly lifted
operator. Refreshing the shared large cache recombines both original-operator
two-tail bases but never rebuilds the shared small basis. Legacy per-arm large
refresh fields are accepted only when their values agree.

The one-tail deflation preconditioners act directly on the current SPD matrix:

```math
P_n=(I-VV^\mathsf{T})
+\tau V(V^\mathsf{T}S_nV)^{-1}V^\mathsf{T},
\qquad \tau>0.
```

Tau selection uses the actual requested basis widths. Define

```math
m_s=\begin{cases}
\texttt{sm\_eig},&\texttt{small\_basis\_source=lanczos},\\
\left\lceil\alpha\,\texttt{sm\_eig}\right\rceil,
&\texttt{small\_basis\_source=inverse\_gaussian},
\end{cases}
\qquad
m_l=\left\lceil\alpha\,\texttt{lg\_eig}\right\rceil,
\quad \alpha=\texttt{sketch\_oversampling}.
```

With sorted eigenvalues of the current Schur matrix,

```math
\lambda_{\rm lo}=\lambda_{m_s+1},\qquad
\lambda_{\rm hi}=\lambda_{n-m_l},\qquad
\tau_\star=\sqrt{\lambda_{\rm lo}\lambda_{\rm hi}}.
```

The standalone large arm uses $\lambda_{\rm hi}$. The sequential design uses
$\lambda_{\rm hi}$ in stage one and $\tau_\star$ in stage two. The one-shot
concatenated deflator and the adaptive post-lift large deflator use
$\tau_\star$.

For the adaptive design, let $V_s$ be the shared small basis and
$\widehat\lambda_s=\lambda_{\min}(V_s^\mathsf{T}S V_s)$. Its lift is

```math
P_{\rm lift}=I+\tau_{\rm lift}^{-1}V_sV_s^\mathsf{T},\qquad
\tau_{\rm lift}=\frac{\widehat\lambda_s}
{\lambda_{\max}(S)-\widehat\lambda_s}.
```

Thus the smallest captured Rayleigh value is mapped exactly to
$\lambda_{\max}(S)$. For an exact invariant small eigenspace, every captured
eigenvalue is multiplied by the same factor $1+\tau_{\rm lift}^{-1}$, so the
lift can create at most `rank(V_s)` eigenvalues above the old spectral maximum.
The second Gaussian basis is constructed from
$P_{\rm lift}^{1/2}SP_{\rm lift}^{1/2}$ and deflates those large modes. Its
sketch width is
$\min(n,\lceil\alpha(\texttt{sm\_eig}+\texttt{lg\_eig})\rceil)$ so it has
room for both the lifted small tail and the original large tail. The
$\tau_\star$ cutoff continues to use $m_s$ and $m_l$ above.

By default `lift_tau=[]`, so the formula above is used dynamically. Setting a
positive scalar overrides it, for example `lift_tau=1e-10`. This is a very
small tau and therefore a very large lift coefficient
`1/lift_tau=1e10`; the selected value is cached with the shared small basis.

The basis lives in the physical coordinates of $S_n$. There is no inner split
No `ichol` or sparse-proxy arm is included because the Schur complement's
matrix representation is dense.

The main benchmark defaults to `EXACT_DENSE_DIAGNOSTICS=false`. Its Schur
drift plots use fixed Gaussian probe actions, and spectral targets use
matrix-free extremal Ritz estimates. Setting the flag to `true` restores exact
Frobenius drift, exact inverse drift, full eigenvalues, and a current Cholesky
check by explicitly materializing every required $S_n$. The dedicated spectrum,
rank, and extraction scripts always request dense matrices because exact dense
quantities are their purpose. `Astat.dense_materialized_step` records every
time step at which the main sequence requested a dense matrix.

Normal time stepping also avoids the former per-step `K\b` factorization. A
tighter-tolerance Schur PCG solve supplies the reference solution and advances
the velocity state. `REFERENCE_TOL` and `REFERENCE_MAXIT` control that solve,
and its iterations, flag, and residual are recorded. Setting
`EXACT_REFERENCE_DIAGNOSTICS=true` explicitly materializes the sparse KKT pair
and runs `K\b` for validation; `Astat.kkt_materialized_step` records those
requests. Exact-only fields remain `NaN` when that option is disabled.

Setting `PLOT_EXTREME_EIGENVALUES=true` enables two additional matrix-free
Ritz estimates for every configured linear system and time step. The estimates
come from each symmetric two-sided preconditioned operator, so they represent
the spectrum relevant to PCG rather than the generally nonsymmetric product
`P*S`. The option is disabled by default because its cost scales with the
number of solver arms. It reuses `SPECTRAL_RITZ_TOL` and
`SPECTRAL_RITZ_MAXIT`. Because every measured operator is SPD, its
smallest-magnitude eigenvalue is recovered from the largest-real Ritz value of
the shifted operator `lambda_max*I-A`. This remains fully matrix-free, makes
the desired mode dominant, and avoids an inner shift-invert solve. Both
extreme eigenpairs receive an operator-norm-relative residual check;
a failed optional estimate is recorded without materializing a dense operator
or aborting the benchmark. The statistics include the two extremes, their
ratio, the maximum Ritz residual, and whether the value was exact.

## 6. Benchmark cases

The Schur study retains the parent's three cases and adds one Schur-local
adversarial case.

| case | solid motion | viscosity field | expected Schur behavior |
|---|---|---|---|
| `bar_rotating_nu_orbiting` | rotating bar | orbiting high-contrast blobs and co-rotating striations | strongest full-operator drift |
| `disk_translating_nu_wake` | translating disk | moving low-viscosity wake | smoother, milder drift |
| `disk_static_nu_checkerboard_shift` | stationary disk | smooth 100:1 checkerboard shifted by half a wavelength | viscosity-only broad generalized-spectrum drift |
| `disk_static_nu_const` | stationary disk | $\nu\equiv1$ | constant-operator control |

## 7. Running the benchmark

Start MATLAB in `symindefinite/stokes_varvisc_rotor_schur_comp` and run:

```matlab
SMOKE_TEST = true; run_varvisc_schur_recycle
run_varvisc_schur_recycle
run_varvisc_schur_spectrum
run_varvisc_schur_rank
varvisc_schur_extract_examples  % default: stress case, h0=0.05, step 1

% Adaptive-deflator tuning (expensive production sweep):
run_varvisc_schur_adaptive_tuning

% Fast workflow check:
run_varvisc_schur_adaptive_tuning(struct('smoke',true))

% Optional single-snapshot configuration, set before invoking the script:
EXTRACT_CASE_NAME = 'disk_translating_nu_wake';
EXTRACT_H0 = 0.1;
EXTRACT_STEP = 2;
EXTRACT_OUTPUT_DIR = tempdir;
varvisc_schur_extract_examples

cd tests
run_all_tests
```

Smoke mode executes three stress-case steps on an `h0=0.1` mesh without
changing `Tstep`, so it preserves the production motion. Full runs use
`h0=0.05`, 60 solves, and all four cases.

Outputs include `all_results.csv`, `speedup_summary.csv`, per-case solver and
operator-drift plots, cross-case summaries, and `run_config.{mat,json}`. When
`PLOT_EXTREME_EIGENVALUES=true`, each case also writes
`plot_smallest_eigenvalues.png`, `plot_largest_eigenvalues.png`, and
`plot_preconditioned_kappa.png`, with one trajectory per configured system.
The master CSV receives matching `<solver>_lambda_min`,
`<solver>_lambda_max`, `<solver>_kappa_prec`, `<solver>_spectrum_flag`,
`<solver>_spectrum_residual`, and `<solver>_spectrum_is_exact` columns, and
`replot_varvisc_schur` recreates all three figures from those columns.
Generated output directories and example `.mat` files are ignored by git.

`run_varvisc_schur_adaptive_tuning` tunes only the existing adaptive knobs on
`disk_static_nu_checkerboard_shift`, while preserving the production
20-small/140-post-lift effective dimensions. It screens dynamic tau, the current
`1e-10` override, and fixed multiples of the step-1 dynamic tau together with
`lift_large_q=[0,1,2]`. Both the small basis and adaptive transformed-large
basis remain frozen with refresh interval `Inf` in every candidate. The coarse
`h0=0.1` stage promotes six candidates to the full
`h0=0.05`, 60-step stage. The winner has the smallest worst-timestep
preconditioned condition number; candidates within one percent use total PCG
iterations as the tie-breaker. CSV files retain both summaries and per-step
spectra, and `recommended_config.json` records the selected existing-knob
settings. This is the best tested configuration for the hard-drift case, not a
claim of a global optimum.

`varvisc_schur_extract_examples` marches exact dense Schur solves through the
requested step and writes one
`varvisc_schur_example_<case>_h<h0>_step<step>.mat` artifact. It contains the
reduced system `S*y_ref = rhs_S`, the complete ordered `eigenvalues`, the
logical pin-removal map `keep`, and validation/configuration `meta`. To rebuild
the full pressure/constraint vector, use:

```matlab
y = zeros(meta.nS_full,1);
y(keep) = y_ref;
y(meta.pin_node) = meta.pin_val;
```

The matching `<artifact-stem>_spectrum.png` plots every eigenvalue on a
logarithmic scale and reports `meta.lambda_min`, `meta.lambda_max`, and
`meta.condition_number`. The extractor uses dense `eig(S)`, so this is the
complete spectrum rather than an iterative estimate. Before writing either
file it checks symmetry, Cholesky success, the Schur residual, and recovered
solution agreement with the parent `K\b` solve.

`run_varvisc_schur_spectrum` additionally writes the actual ordered eigenvalue
curves to `spectrum/spectrum_raw_vs_prec.png` and
`spectrum/spectrum_raw_snapshots.png`, alongside CSV spectrum summaries and
conditioning plots.

## 8. Verification and code map

The test suite checks the defining Schur properties:

- `test_varvisc_schur_correctness` compares Schur recovery with full `K\b`,
  compares vector and block operator products with the explicit matrix, checks
  symmetry and Cholesky success, and verifies velocity boundary values;
- `test_varvisc_schur_pin` verifies the decoupled `-1` pressure-pin direction
  and its deletion;
- `test_varvisc_schur_structure` verifies motion of $A_n$, $D_n$, and the
  pressure-pressure Schur block, as well as failure of the old rank bound;
- `test_varvisc_schur_drift` distinguishes a stale frozen inverse in moving
  cases from the exact frozen inverse in the static control;
- `test_varvisc_schur_projector` checks the Gaussian basis and the symmetry,
  positive definiteness, spectral action, and absence of post-`orth` column
  removal in the deflation preconditioner; and
- `test_varvisc_schur_hard_case` verifies viscosity-only complementary drift,
  broad generalized spectral damage, and strong recycled-Cholesky iteration
  growth;
- `test_varvisc_schur_extreme_eigenvalues` validates the symmetric
  preconditioned extrema, residuals, CSV columns, and plots; and
- `test_varvisc_schur_adaptive_tuning` exercises the two-stage workflow and
  its recommendation artifacts in smoke mode;
- `test_varvisc_schur_gaussian_refresh` verifies the paired inverse-power
  bases, current-factor freshness, fixed tau, and recovered KKT solution;
- `test_varvisc_schur_augmentation` checks projected Arnoldi against an explicit
  projection, coarse SPD, exact `m=0` equivalence, frozen-cache behavior,
  operator counts, breakdown cases, and direct-solve agreement; and
- `test_varvisc_schur_augmented_workflow` checks checkpoint resumption,
  rejection of incomplete finalization, and configuration mismatch handling; and
- `test_varvisc_schur_two_stage_augmentation` checks the shared augmented basis
  in both stages, warm-start residuals, zero-augmentation equivalence, refreshes,
  independent configuration, SPD/spectral correctness, CSVs and replotting.

The construction is split between four benchmark-local helpers:

- `varvisc_schur_assemble_blocks.m` directly assembles the post-boundary sparse
  blocks and stores both orientations of the coupling;
- `varvisc_schur_assemble_kkt.m` retains the monolithic exact-reference
  assembly used by cross-checks;
- `varvisc_schur_context_init.m` stores only time-independent mesh and assembly
  data; and
- `varvisc_schur_step_operator.m` removes the pin, exposes `apply`, `to_dense`,
  and lazy `materialize_kkt` handles, and defines full-solution recovery.

All benchmark-local helpers use the `varvisc_schur` prefix to avoid name
collisions with the KKT and constant-viscosity Schur benchmarks.
