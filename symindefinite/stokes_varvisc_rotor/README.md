# Variable-viscosity Stokes immersed rotor

This benchmark is the variable-viscosity sibling of
[`stokes_immersed_rotor`](../stokes_immersed_rotor/README.md). It solves
backward-Euler, incompressible Stokes flow in the channel
$\Omega=[0,4]\times[0,1]$ with two independent sources of operator motion:

1. a moving rigid solid, imposed on a fixed fluid mesh by distributed Lagrange
   multipliers; and
2. a moving, high-contrast element viscosity $\nu(x,t)$, representing a
   low-viscosity phase stirred by the rotor.

Every time step produces a **symmetric indefinite saddle-point (KKT) system**.
In the parent benchmark only the immersed coupling border changes, giving a
structured low-rank update. Here the viscosity-weighted velocity and pressure
blocks also change throughout their sparsity patterns. That difference removes
the parent's low-rank safety net and is the point of this benchmark. The
immersed-solid formulation follows the same simplified MATLAB port of
[deal.II step-70](https://dealii.org/current/doxygen/deal.II/step_70.html) as
the parent.

## 1. Physical problem

Let $u=(u_x,u_y)$ be the fluid velocity and $p$ the pressure. A rigid bar or
disk occupies the moving region $B(t)\subset\Omega$ and has prescribed rigid
velocity $g(t)$. The fluid is required to match that velocity at Lagrange
points carried by the solid. The multiplier $\lambda$ is the distributed
reaction force needed to enforce the match.

The benchmark uses the componentwise variable-viscosity Stokes model

```math
\partial_t u-\nabla\!\cdot\!\big(\nu(x,t)\nabla u\big)+\nabla p=f
\qquad\text{in }\Omega,
```

```math
\nabla\!\cdot u=0
\qquad\text{in }\Omega,
```

```math
u=g(t)
\qquad\text{on }B(t)\quad\text{(enforced weakly through }\lambda\text{)}.
```

The moving coefficient is sampled once per element, at the element centroid.
The stress case ranges from $\nu_{\min}=0.02$ to $\nu_{\max}=2$, a 100:1
contrast. Regions of small $\nu$ diffuse momentum less strongly; as those
regions orbit with the rotor, the local stiffness contributions change across
the channel even though the background mesh is fixed.

**Boundary conditions** on the channel boundary $\partial\Omega$ are the same
as in the parent benchmark:

- parabolic inflow on the left,
  $u_x=U_{\mathrm{in}}4y(1-y)$ and $u_y=0$;
- no slip, $u=0$, on the top and bottom walls;
- natural (do-nothing) outflow on the right; and
- one pressure value pinned at an outflow node to remove the constant-pressure
  null space.

The immersed condition on $B(t)$ is separate from these outer-boundary
conditions: the solid does not cut a hole in or deform the fluid mesh.

## 2. Weak formulation

Let $V$ be the velocity space, $Q$ the pressure space, and $\Lambda(t)$ the
multiplier space supported on the moving solid. Before pressure stabilization,
the variational problem is to find $(u,p,\lambda)\in V\times Q\times\Lambda(t)$
such that, for all $(v,q,\mu)$ in the corresponding test spaces,

```math
\begin{aligned}
(\partial_tu,v)_\Omega
+(\nu\nabla u,\nabla v)_\Omega
-(p,\nabla\!\cdot v)_\Omega
+\langle\lambda,v\rangle_{B(t)}&=(f,v)_\Omega,\\
(\nabla\!\cdot u,q)_\Omega&=0,\\
\langle\mu,u\rangle_{B(t)}&=\langle\mu,g(t)\rangle_{B(t)}.
\end{aligned}
```

Pressure and immersed-body motion are two different constraints, enforced by
two different multipliers:

- $p$ enforces incompressibility;
- $\lambda$ enforces the rigid-body velocity.

The implementation uses a symmetric $+B^\mathsf{T}/+B$ block convention. This
amounts to an immaterial sign choice for the discrete pressure relative to the
continuous weak form above; velocity and all constraint equations are
unchanged.

## 3. P1-P1 spatial discretization

The channel is triangulated once and never remeshed. Continuous piecewise-linear
(P1) basis functions are used for both velocity and pressure. If the mesh has
$N$ nodes, the unknown ordering is

```math
x^n=[\,u_x^n;\,u_y^n;\,p^n;\,\lambda^n\,],
```

with $2N$ velocity unknowns, $N$ pressure unknowns, and $n_C(t_n)$ multiplier
unknowns.

### Constant mass and divergence blocks

Let $\{\phi_i\}$ be the scalar P1 nodal basis. The scalar consistent mass
matrix is

```math
D_{ij}=\int_\Omega\phi_i\phi_j,
\qquad
D_e=\frac{|e|}{12}
\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix},
```

and the two-component velocity mass is $M_2=\operatorname{blkdiag}(D,D)$.

The discrete divergence $B\in\mathbb R^{N\times2N}$ is

```math
B_{k,(j,c)}=\int_\Omega\phi_k\,\partial_{x_c}\phi_j,
\qquad c\in\{x,y\}.
```

Both $M_2$ and $B$ depend only on the fixed mesh and are assembled once by
`assemble_stokes_blocks`.

### Viscosity-weighted velocity stiffness

At step $n$, `nu_fun` evaluates one positive value $\nu_e(t_n)$ at every
element centroid. `assemble_visc_stiffness` rescales the stored unit-stiffness
triplets to assemble

```math
(K_\nu^n)_{ij}
=\sum_{e\in\mathcal T_h}\nu_e(t_n)
\int_e\nabla\phi_i\!\cdot\!\nabla\phi_j,
```

and the vector diffusion block is

```math
A_2(\nu^n)=\operatorname{blkdiag}(K_\nu^n,K_\nu^n).
```

Unlike the constant-viscosity parent, $K_\nu^n$ and therefore $A_2(\nu^n)$
are reassembled at every step.

### Equal-order pressure stabilization

Equal-order P1-P1 velocity-pressure elements do not satisfy the discrete
inf-sup condition without stabilization. The benchmark uses an elementwise
Brezzi-Pitkäranta coefficient

```math
\varepsilon_e^n=\frac{h_0^2}{12\,\nu_e(t_n)}
```

and assembles

```math
(L_{p,\varepsilon}^n)_{ij}
=\sum_{e\in\mathcal T_h}\varepsilon_e^n
\int_e\nabla\phi_i\!\cdot\!\nabla\phi_j.
```

The KKT system contains $-L_{p,\varepsilon}^n$ in the pressure block. Thus the
pressure stabilization changes whenever the viscosity changes, and it moves
in the inverse direction: low-viscosity elements receive stronger
stabilization. The engine also supports a scalar fallback, but the benchmark
uses this elementwise form.

### Moving immersed coupling

At each Lagrange point $X_k(t_n)$, the solid constraint is

```math
u_h(X_k(t_n))=V_k(t_n).
```

`assemble_coupling` locates the host triangle and evaluates its three
barycentric weights. Each scalar constraint row consequently has three
nonzeros. Stacking the $x$- and $y$-component constraints gives

```math
C(t_n)u^n=g(t_n),
\qquad
n_C(t_n)=2\times(\text{number of in-domain Lagrange points}).
```

As the solid moves across the fixed mesh, host triangles and interpolation
weights change. Points outside the channel are dropped, so in the general
engine $n_C$ and the system dimension may also change.

## 4. Backward Euler and the per-step linear system

Backward Euler at $t_n=n\Delta t$ gives

```math
\partial_tu(t_n)\approx\frac{u^n-u^{n-1}}{\Delta t}.
```

After spatial assembly, step $n$ solves

```math
\underbrace{\begin{bmatrix}
\dfrac{M_2}{\Delta t}+A_2(\nu^n)
    &B^\mathsf{T}&C(t_n)^\mathsf{T}\\[6pt]
B&-L_{p,\varepsilon}^n&0\\[2pt]
C(t_n)&0&0
\end{bmatrix}}_{\mathcal K_n}
\begin{bmatrix}u^n\\p^n\\\lambda^n\end{bmatrix}
=
\begin{bmatrix}
\dfrac{M_2}{\Delta t}u^{n-1}+M_2f^n\\[6pt]
0\\[2pt]
g(t_n)
\end{bmatrix}.
```

The production cases set $f=0$, but the engine retains the optional body-force
term shown above. The default time step is $\Delta t=0.02$, with 60 solves from
61 time levels.

**Why the system is symmetric.** The off-diagonal blocks occur in transpose
pairs, and all diagonal blocks are symmetric. Velocity Dirichlet values and the
pressure pin are imposed by lifting known values to the right-hand side and
zeroing both the corresponding row and column, so the eliminated system remains
symmetric.

**Why the system is indefinite.** The velocity block is positive definite after
the boundary conditions are applied, while the pressure block is negative
semidefinite and the multiplier diagonal block is zero. The spectrum therefore
contains both positive and negative eigenvalues. MINRES is appropriate for the
symmetric system when supplied with an SPD preconditioner; CG is not.

## 5. How the linear system changes

The time integrator creates the sequence

```math
\mathcal K_nx^n=b_n,
\qquad n=1,\ldots,60,
```

and advances the next right-hand side with the backslash reference velocity
$u^n$.

### What stays constant

The following are built once and reused:

- the mesh, its element geometry, and the `triangulation` search object;
- the velocity mass $M_2$;
- the divergence pair $B$ and $B^\mathsf{T}$;
- the boundary-node sets and pressure-pin location; and
- the unit-stiffness triplets used to assemble coefficient-weighted matrices.

### What changes at step n

- **Element viscosity $\nu_e(t_n)$.** Moving blobs and striations change the
  coefficient sampled on the elements.
- **Velocity block.** $A_2(\nu^n)$ is reassembled, so
  $M_2/\Delta t+A_2(\nu^n)$ changes throughout its existing sparsity pattern.
- **Pressure block.** Because $\varepsilon_e^n=h_0^2/(12\nu_e^n)$,
  $L_{p,\varepsilon}^n$ is also reassembled.
- **Immersed border and constraint data.** Motion changes $C(t_n)$ and $g(t_n)$.
- **Right-hand side.** It carries $u^{n-1}$ forward and includes any changing
  body force, rigid velocity, or prescribed boundary data.
- **Preconditioner data.** The current boundary-eliminated velocity block,
  viscosity-weighted pressure-mass diagonal, ILDL factors, and split
  coordinates may all change according to their refresh cadences.

To isolate fluid-block drift, define

```math
F_n=
\begin{bmatrix}
M_2/\Delta t+A_2(\nu^n)&B^\mathsf{T}\\
B&-L_{p,\varepsilon}^n
\end{bmatrix}.
```

Then the full update contains two qualitatively different parts:

```math
\mathcal K_n-\mathcal K_{n-1}
=
\begin{bmatrix}
F_n-F_{n-1}&\Delta\widetilde C_n^\mathsf{T}\\
\Delta\widetilde C_n&0
\end{bmatrix},
```

where $\widetilde C_n=[\,C(t_n)\;0_p\,]$ embeds the velocity coupling into the
fluid $(u,p)$ columns. This expression assumes equal constraint counts at the
two steps; otherwise even the matrix dimension changes.

### Contrast with `stokes_immersed_rotor`

For the constant-viscosity parent, $F_n=F_1$ and only the coupling border moves:

```math
\mathcal K_n-\mathcal K_1
=
\begin{bmatrix}
0&\Delta\widetilde C_n^\mathsf{T}\\
\Delta\widetilde C_n&0
\end{bmatrix},
\qquad
\operatorname{rank}(\mathcal K_n-\mathcal K_1)\le 2n_C.
```

That low-rank identity supports Woodbury formulas and the parent's
$2n_C+1$ unrestarted-GMRES finite-termination bound with the exact frozen
inverse.

Here $F_n-F_1\ne0$ on the moving-viscosity cases. The stiffness difference
touches nearly every nonzero in the velocity-block pattern
(`dK_nnz_frac` is about 0.998 in the stress case), and the singular-value test
in `varvisc_convergence_test` requires $r_{90}=O(N)$ modes to capture 90% of
its Frobenius norm. The update is therefore numerically full-rank at benchmark
scale: there is no $2n_C$ rank ceiling, no corresponding Woodbury shortcut, and
no $2n_C+1$ GMRES guarantee.

The `disk_static_nu_const` case is the falsification control. Its solid is
stationary and $\nu\equiv1$, so $F_n$, $C(t_n)$, $\mathcal K_n$, and the
boundary data are constant. Frozen and refreshed preconditioners must therefore
coincide.

### Recorded change diagnostics

The engine records complementary measurements rather than folding every change
into one number:

```math
\mathtt{diffK}_n
=\frac{\|F_n-F_{n-1}\|_F}{\|F_{n-1}\|_F},
\qquad
\mathtt{coupling\_change}_n
=\frac{\|C_n-C_{n-1}\|_F}{\|C_{n-1}\|_F},
```

```math
\mathtt{dK\_nnz\_frac}_n
=\frac{\operatorname{nnz}(A_{vel}^n-A_{vel}^{n-1})}
       {\operatorname{nnz}(A_{vel}^n)},
\qquad
\mathtt{nu\_contrast}_n
=\frac{\max_e\nu_e^n}{\min_e\nu_e^n}.
```

`diffK` deliberately excludes $C$ so fluid-coefficient drift and immersed-body
motion can be read separately. `coupling_change` is reported only when adjacent
coupling matrices have compatible row counts.

## 6. Benchmark cases

The cases are defined in `varvisc_define_case_list.m`.

| case | solid motion | viscosity field | purpose |
|---|---|---|---|
| `bar_rotating_nu_orbiting` | rotating bar | two orbiting log-Gaussian blobs at the bar tips plus co-rotating striations, $\nu\in[0.02,2]$ | 100:1 full-rank-drift stress case |
| `disk_translating_nu_wake` | translating disk | one trailing low-viscosity wake blob, $\nu\in[0.04,2]$ | 50:1 smooth moving-coefficient case |
| `disk_static_nu_const` | static disk | $\nu\equiv1$ | constant-operator control |

The striations in the stress case are deliberate. Smooth blobs alone yield a
numerically compressible stiffness difference; the rotating sign-oscillatory
texture distributes the update energy over $O(N)$ singular directions.

## 7. Solver registry and preconditioners

Each step is solved first by sparse backslash, which supplies the ground truth
and advances the state, then by one Krylov solve per entry in
`varvisc_define_solver_list.m`. All entries use MINRES except the exact-signed-
inverse GMRES arm. There is intentionally **no Krylov-subspace-recycling arm**.

| key | preconditioner or solve |
|---|---|
| `minres_unprec` | none |
| `block_jacobi` | $\operatorname{blkdiag}(A_{vel},\operatorname{diag}(m/\nu),I_\lambda)$ approximation, rebuilt every step |
| `block_jacobi_frozen` | the same block preconditioner, frozen at step 1 |
| `ildl_nofill` | incomplete-LDL split solve, no coarse space |
| `exact_ldl_frozen` | exact LDL of $\mathcal K_1$, SPD-ified as $|\mathcal K_1|$, then frozen |
| `two_level_sjlt` / `two_level_gaussian` | ILDL plus deflation using a randomized coarse basis |
| `two_level_polynomial` / `two_level_exact` | ILDL plus a Chebyshev high-pass or exact-`eigs` coarse basis |
| `gmres_exact_inv_frozen` | unrestarted GMRES with the exact signed $\mathcal K_1^{-1}$ |
| `two_level_esketch` | ILDL plus a randomized eigen-sketch of the scaled update $E$ |

The block-Jacobi comparison is the simplest view of coefficient drift. The
refreshed arm rebuilds its incomplete Cholesky factor and viscosity-weighted
pressure diagonal from the current operator; the frozen arm keeps both from
step 1. Their gap should open on moving-viscosity cases and vanish on the
control.

Deflation bases cached across steps are stored in **physical coordinates** and
transported into the current split coordinates as
$V_n=\operatorname{orth}(C_n^\mathsf{T}U)$. This transport is required when the
ILDL factor is refreshed: reusing the same coordinate array would represent a
different physical subspace. It is coordinate transport, not Krylov recycling.

### Unified randomized-sketch parameters

The Gaussian, SJLT, and update-$E$ sketches use one configuration:

- width `SKETCH_OVERSAMPLE * DEFLAT_SM_EIG` (default $2\times500=1000$);
- `DEFLAT_Q` power-iteration rounds (default 2); and
- no requested-rank truncation: the final basis is only orthonormalized, so its
  dimension is the numerical rank of the full oversampled sketch.

The deterministic `two_level_exact` and `two_level_polynomial` bases retain
dimension `DEFLAT_SM_EIG` (500). Randomized arms therefore carry a coarse space
about twice as large by default.

The $E$-sketch targets the symmetric operator

```math
E=C_1^{-1}(\mathcal K_n-\mathcal K_1)C_1^{-\mathsf T},
```

where $C_1$ is the exact step-1 split factor. Since
$C_1^{-1}\mathcal K_nC_1^{-\mathsf T}=\operatorname{sign}(D_1)+E$, the sketch
uses the power iteration $Y=E^{2q+1}\Omega$ to approximate dominant
eigendirections of the perturbation actually seen by split MINRES. Its cost is
reported in operation counts: each application uses one sparse
$\Delta\mathcal K$ multiplication and triangular solves with $C_1$ and
$C_1^\mathsf T$, batched across sketch columns.

On the constant control, $\Delta\mathcal K=0$, so the $E$-sketch correctly
returns an empty coarse basis and reduces to plain ILDL.

## 8. Running the benchmark

Start MATLAB in `symindefinite/stokes_varvisc_rotor` (or change to that folder)
and run:

```matlab
run_varvisc_benchmark                       % full 3-case, 60-step benchmark
SMOKE_TEST = true; run_varvisc_benchmark    % 2 solves, stress case only
run_varvisc_spectrum_spy                    % spectra at five time levels
SMOKE_TEST = true; run_varvisc_spectrum_spy % reduced end-to-end spectrum check
varvisc_convergence_test                    % MMS, KKT, motion, and rank gates
replot_varvisc_benchmark                    % redraw from saved CSV/config data
```

The main run writes `benchmark_varvisc/`:

```text
benchmark_varvisc/
  all_results.csv
  speedup_summary.csv
  paper_summary_table.csv
  run_config.mat
  run_config.json
  summary_plots/
  iteration_vs_timestep/
  <case>/
    <key>_solver_iterations.csv
    <key>_solver_iterations.png
    all_solvers_comparison.png
    relative_step_to_step_change.png
    accuracy.png
    coefficient_movie/              # stress case only
```

`all_results.csv` contains one row per `(case, step)`, `<key>_its` and
`<key>_flag` columns for every registry entry, accuracy information, and the
change metrics described above.

`run_varvisc_spectrum_spy` writes `output/spectrum_spy_varvisc/`. At steps 1,
15, 30, 45, and 60 of the stress case it compares:

1. the original KKT matrix $\mathcal K_n$;
2. the frozen exact-LDL split operator
   $C_1^{-1}\mathcal K_nC_1^{-\mathsf T}$; and
3. its scaled update $C_1^{-1}(\mathcal K_n-\mathcal K_1)C_1^{-\mathsf T}$.

It writes lower- and upper-spectral-tail figures, first/last sparsity and sign
patterns, and `spectrum_summary.csv`. Exact zeros in the structurally singular
update are displayed in gray at a labeled logarithmic plotting floor.

## 9. Expected behavior

- On `bar_rotating_nu_orbiting`, frozen block Jacobi, exact LDL, and exact
  inverse should deteriorate as the viscosity field moves; refreshed block
  Jacobi should remain comparatively flat.
- On `disk_translating_nu_wake`, the same effect should be present but milder
  because the coefficient field is smoother.
- On `disk_static_nu_const`, frozen and refreshed curves should coincide; exact
  frozen LDL remains exact, and the $E$-sketch reduces to ILDL.
- Unpreconditioned MINRES may reach `SOLVER_MAXIT`. This is a benchmark result.
  Nonzero flags from the intended preconditioned arms warrant investigation.
- `gmres_exact_inv_frozen` may reach `GMRES_MAXIT` on moving-viscosity cases.
  Unlike the parent benchmark, no low-rank finite-termination theorem applies.

## 10. Verification

`varvisc_convergence_test` checks the discretization and the claimed operator
motion:

- manufactured-solution spatial convergence for velocity and pressure;
- first-order backward-Euler temporal convergence with time-varying viscosity;
- KKT symmetry to roundoff and eigenvalues of both signs;
- rigid-body constraint satisfaction;
- nontrivial viscosity contrast, fluid-block change, and coupling change in the
  stress case; and
- a singular-value rank certificate showing that the viscosity-induced
  stiffness update is not bounded by the parent's coupling-border rank.

## 11. Code map

- Engine: `+src/+stokes/solve_stokes_varvisc.m`.
- Coefficient assembly: `+src/+stokes/assemble_visc_stiffness.m`.
- Shared mass, divergence, coupling, and symmetric boundary elimination:
  `+src/+stokes/assemble_stokes_blocks.m`, `assemble_coupling.m`, and
  `apply_dirichlet_sym.m`.
- Benchmark cases and solver registry: `varvisc_define_case_list.m` and
  `varvisc_define_solver_list.m`.
- Preconditioner kernels: `+src/+precond/*`.
- Coordinate transport: `../linear_solves/subspace_recycle/kernel`.

All benchmark-local helpers use the `varvisc` prefix so they neither shadow nor
are shadowed by helpers from `stokes_immersed_rotor` when both folders are on the
MATLAB path.
