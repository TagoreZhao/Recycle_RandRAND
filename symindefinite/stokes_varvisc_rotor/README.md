# stokes_varvisc_rotor — variable-viscosity Stokes with an immersed rotor

Backward-Euler unsteady **variable-viscosity** Stokes in a 2D channel
$\Omega = [0,4]\times[0,1]$ with a moving immersed rigid solid enforced by
distributed Lagrange multipliers (deal.II step-70 lineage), plus a moving
high-contrast viscosity field $\nu(x,t)$ (up to 100:1) representing a
low-viscosity phase stirred by the rotor:

$$
u_t - \nabla\cdot(\nu(x,t)\nabla u) + \nabla p = f,\qquad
\nabla\cdot u = 0,\qquad u = g(t)\ \text{on}\ B(t).
$$

Equal-order P1–P1 with Brezzi–Pitkäranta stabilization
($\varepsilon_e = h_0^2/(12\,\nu_e)$, elementwise), backward Euler
($\Delta t = 0.02$, 60 steps), DOF order $[u_x;\,u_y;\,p;\,\lambda]$.
Per step $n$ the symmetric indefinite KKT system is

$$
\begin{bmatrix}
M_2/\Delta t + A_2(\nu_e) & B^{\mathrm T} & C(t_n)^{\mathrm T}\\
B & -L_{p,\varepsilon}(\nu_e) & 0\\
C(t_n) & 0 & 0
\end{bmatrix}
\begin{bmatrix} u\\ p\\ \lambda \end{bmatrix}
=
\begin{bmatrix} (M_2/\Delta t)\,u^{n-1}\\ 0\\ g(t_n) \end{bmatrix}.
$$

## Why this benchmark exists

In the parent `stokes_immersed_rotor` only the coupling border $C(t)$ moves,
so $K_n - K_1$ has rank $\le 2 n_C$ and frozen factorizations enjoy a
structural safety net (Woodbury shortcuts, the GMRES $2n_C{+}1$
finite-termination bound, sketches that capture the *whole* update).  Here
the moving $\nu$ field rebuilds $A_2(\nu_e)$ and $L_{p,\varepsilon}(\nu_e)$
at **every nonzero, every step**: the per-step update is dense in the
sparsity pattern and numerically full-rank (`dK_nnz_frac ≈ 0.998`,
$r_{90} = O(N)$, certified by `varvisc_convergence_test.m`).  Every parent
preconditioner arm is re-measured with that safety net removed.

## Cases (`varvisc_define_case_list.m`)

| case | solid motion | viscosity field |
|---|---|---|
| `bar_rotating_nu_orbiting` (stress) | rotating bar | two orbiting log-Gaussian blobs at the bar tips + co-rotating striations, $\nu\in[0.02,2]$ (100:1) |
| `disk_translating_nu_wake` | translating disk | single trailing wake blob, $\nu\in[0.04,2]$ (50:1) |
| `disk_static_nu_const` (control) | static disk | $\nu\equiv 1$ — $K(t)$ constant, frozen $\equiv$ refreshed |

## Solver registry (`varvisc_define_solver_list.m`)

Per step: backslash (ground truth, advances the state) + one Krylov solve per
entry.  All MINRES except the GMRES arm (its preconditioner is indefinite by
construction).  **No Krylov-subspace-recycling arm.**

| key | preconditioner |
|---|---|
| `minres_unprec` | none |
| `block_jacobi` | $\mathrm{blkdiag}(\mathrm{ichol}(A_{vel}),\ \mathrm{diag}(m/\nu),\ I)$, rebuilt every step |
| `block_jacobi_frozen` | same, frozen at step 1 — the frozen-vs-refreshed gap is the headline diagnostic |
| `ildl_nofill` | incomplete-LDL split solve, no coarse space |
| `exact_ldl_frozen` | exact LDL of $K_1$, SPD-ified ($M=|K_1|$), frozen |
| `two_level_sjlt` / `two_level_gaussian` | ILDL + deflation $L^{-\mathrm T}PL^{-1}$, randomized sketch V |
| `two_level_polynomial` / `two_level_exact` | same scheme, Chebyshev high-pass / exact `eigs` V (deterministic, dim `DEFLAT_SM_EIG`) |
| `gmres_exact_inv_frozen` | exact signed $K_1^{-1}$, frozen; no finite-termination bound here — hitting `GMRES_MAXIT` on the moving-$\nu$ cases is the expected negative result |
| `two_level_esketch` | ILDL + deflation, V = randomized eigen-sketch of the symmetric $E = C_1^{-1}(K_n-K_1)C_1^{-\mathrm T}$, $C_1$ the exact step-1 split factor (the frozen *factorization* reused, not a subspace) |

Deflation bases are cached across steps in **physical coordinates** and
transported into each step's split coordinates on use
(`V_n = \mathrm{orth}(C_n^{\mathrm T} U)`); with `ILDL_PREC_REFRESH = 1` this
coordinate transport is mandatory, and it is *not* Krylov recycling.

## Unified sketch parameters

Every randomized sketch — gaussian V, sjlt V, and the update $E$-sketch —
uses ONE configuration:

- width `SKETCH_OVERSAMPLE * DEFLAT_SM_EIG` (default 2 × 500 = 1000 columns),
- `DEFLAT_Q` power-iteration rounds (default 2),
- **no truncation**: the basis is only orthonormalized at the end, so its
  dimension is the numerical rank of the full oversampled sketch.

The former `LOWRANK_SM_EIG` / `LOWRANK_OVERSAMPLE` / `LOWRANK_SKETCH_Q`
knobs are gone.  Note the deliberate asymmetry: the deterministic
`two_level_exact` / `two_level_polynomial` bases keep dimension
`DEFLAT_SM_EIG` (500), so randomized arms carry a coarse space about twice
as large.

The $E$-sketch targets the operator that actually perturbs the split system
MINRES runs on: with the exact step-1 factor, $C_1^{-1}K_nC_1^{-\mathrm T} =
\mathrm{sign}(D) + E$.  $E$ is symmetric, so the sketch is the plain power
iteration $Y = E^{2q+1}\Omega$ (dominant *eigenvectors*), and the basis is
transported into step $n$'s coordinates by the similarity map
$C_n^{\mathrm T}C_1^{-\mathrm T}$ — which preserves eigenspaces exactly,
unlike the singular spaces of its predecessor, the physical-coordinate
$D = K_1^{-1}(K_n-K_1)$ sketch it replaced.  Sketch cost is reported in
operation counts ($(2q{+}1)\cdot k$ applications of $E$, each one sparse
$\Delta K$ matvec plus one triangular solve against each of $C_1$ and
$C_1^{\mathrm T}$, all batched), never wall-clock.

## Running

Start MATLAB in `symindefinite/stokes_varvisc_rotor` (or `cd` there first),
then run:

```matlab
run_varvisc_benchmark            % full 3-case, 60-step run -> benchmark_varvisc/
SMOKE_TEST = true; run_varvisc_benchmark   % 2 steps, stress case only
run_varvisc_spectrum_spy         % K_n vs recycled exact-LDL spectra at 5 times
SMOKE_TEST = true; run_varvisc_spectrum_spy % reduced spectrum end-to-end check
varvisc_convergence_test         % MMS orders + KKT gates + full-rank certification
replot_varvisc_benchmark         % redraw all figures from all_results.csv
```

Outputs: `all_results.csv` (per-(case,step) row; `<key>_its`/`<key>_flag`
per registry entry plus `diffK`, `nu_contrast`, `dK_nnz_frac`),
`speedup_summary.csv`, `paper_summary_table.csv`, per-case figure dirs,
`summary_plots/`, `iteration_vs_timestep/`, `run_config.{mat,json}`.

`run_varvisc_spectrum_spy` writes `output/spectrum_spy_varvisc/`: smallest-
and largest-magnitude spectral-tail figures with three rows—the original KKT
matrix, the recycled exact-LDL split operator $C_1^{-1}K_nC_1^{-\mathrm T}$,
and its scaled update $C_1^{-1}(K_n-K_1)C_1^{-\mathrm T}$. The exact LDL factor
$C_1$ is built at step 1 and recycled unchanged. Because the update is
structurally singular, its literal lower tail includes exact zeros; these are
shown in gray at a labeled logarithmic plotting floor. The script also writes
first/last sparsity and sign patterns plus `spectrum_summary.csv` for steps 1,
15, 30, 45, and 60 of the rotating-bar stress case. Update lower-tail analysis
forms the sparse reference metric $C_1C_1^{\mathrm T}$ and is therefore more
expensive than the matrix-free upper-tail calculation.

## Expected behavior

- Stress case: frozen arms (`block_jacobi_frozen`, `exact_ldl_frozen`,
  `gmres_exact_inv_frozen`) degrade as $t$ grows; refreshed `block_jacobi`
  stays flat.
- Unpreconditioned MINRES may reach `SOLVER_MAXIT`; this is a benchmark result,
  while nonzero flags in the preconditioned arms warrant investigation.
- Control case: frozen $\equiv$ refreshed (curves coincide), and the
  $E$-sketch arm degrades to plain ILDL ($\Delta K = 0$ returns an empty V —
  the honest answer, not an error).

## Shared code

Engine `+src/+stokes/solve_stokes_varvisc.m` (preconditioner-agnostic, same
registry contract as `solve_stokes_immersed`), assembly
`+src/+stokes/assemble_visc_stiffness.m`, preconditioners `+src/+precond/*`,
coordinate transport from
`symindefinite/linear_solves/subspace_recycle/kernel` (`transport_V`,
`ildl_coordinate_map`, `orth_trunc`).  All benchmark-local helpers carry the
`varvisc` prefix so nothing shadows (or is shadowed by) the parent
`stokes_immersed_rotor` helpers when both are on the path.
