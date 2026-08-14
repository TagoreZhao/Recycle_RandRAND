# Schur complement of variable-viscosity immersed-rotor Stokes

This benchmark is the reduced-system counterpart of
[`../stokes_varvisc_rotor/`](../stokes_varvisc_rotor/). It asks how a recycled
dense Schur-complement solver behaves when the viscosity field changes the
fluid operator throughout its sparsity pattern, rather than only moving the
immersed-boundary coupling border.

## Reduced system

At step `n`, after symmetric velocity boundary elimination and pressure
pinning, write the KKT system as

\[
\begin{bmatrix}A_n&G_n^T\\G_n&-D_n\end{bmatrix}
\begin{bmatrix}u\\y\end{bmatrix}=
\begin{bmatrix}b_1\\b_2\end{bmatrix},\qquad y=(p,\lambda).
\]

Eliminating velocity gives

\[
S_n=D_n+G_nA_n^{-1}G_n^T,\qquad
S_ny=G_nA_n^{-1}b_1-b_2,
\]

followed by `u = A_n \ (b1-G_n'*y)`. The pressure pin inserted by
`apply_dirichlet_sym` is a decoupled `-1` direction in the unreduced Schur
matrix. It is deleted before PCG and scattered back during recovery. The
remaining dense matrix is SPD.

## What variable viscosity changes

In `stokes_immersed_rotor_schur_comp`, the velocity block and the
pressure-pressure Schur block are constant, so only `nC` velocity backsolves
are required per step and `rank(S_n-S_m) <= 2*nC`.

That shortcut is invalid here. Elementwise viscosity changes both

- `A_n = M/dt + A_2(nu_n)`, and
- `D_n`, through `eps_e = h0^2/(12*nu_e)`.

Consequently each step rebuilds `chol(A_n)` and forms the complete dense
`S_n`. `run_varvisc_schur_rank` verifies that the pressure-pressure block moves
and that the old border-only rank bound is exceeded. The
`disk_static_nu_const` case is the negative control: all operator blocks remain
constant.

## Solver arms

All arms use the same scaled RHS and warm start.

| key | method |
|---|---|
| `pcg_unprec` | unpreconditioned PCG |
| `chol` | exact dense `chol(S_1)`, frozen for the sequence |
| `deflate_exact` | exact 20-dimensional smallest-eigenvector basis of `S_1`, reused in physical coordinates |
| `deflate_gaussian` | 20-column Gaussian inverse-power sketch of `S_1^{-1}`, reused in physical coordinates |

The deflation preconditioner is built directly on the current SPD matrix:

\[
P_n=(I-VV^T)+\tau V(V^TS_nV)^{-1}V^T,
\qquad \tau=\lambda_{\max}(S_1).
\]

The basis is frozen, but its small Galerkin matrix uses the current `S_n`.
There is no split factor or coordinate transport. No `ichol` or sparse-proxy
arm is included because the exact Schur complement is dense.

## Running

```matlab
cd symindefinite/stokes_varvisc_rotor_schur_comp
SMOKE_TEST = true; run_varvisc_schur_recycle
run_varvisc_schur_recycle
run_varvisc_schur_spectrum
run_varvisc_schur_rank
varvisc_schur_extract_examples

cd tests
run_all_tests
```

Smoke mode executes three stress-case steps on an `h0=0.1` mesh without
changing `Tstep`, so it preserves the production motion. Full runs use
`h0=0.05`, 60 solves, and all three variable-viscosity cases.

Outputs include `all_results.csv`, `speedup_summary.csv`, per-case solver and
operator-drift plots, cross-case summaries, and `run_config.{mat,json}`.
Generated output directories and example `.mat` files are ignored by git.

`run_varvisc_schur_spectrum` additionally writes the actual ordered eigenvalue
curves to `spectrum/spectrum_raw_vs_prec.png` and
`spectrum/spectrum_raw_snapshots.png`, alongside the CSV spectra summaries and
conditioning plots.
