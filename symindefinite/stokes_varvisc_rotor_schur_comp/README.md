# Schur complement of variable-viscosity immersed-rotor Stokes

This benchmark is the reduced-system counterpart of
[`../stokes_varvisc_rotor/`](../stokes_varvisc_rotor/). The parent problem
produces a sequence of sparse, symmetric indefinite Stokes KKT systems. This
benchmark eliminates velocity and instead solves the explicit dense Schur
complement in the pressure and immersed-multiplier variables.

The distinction matters because the reduced operator has different algebraic
properties from the KKT matrix: after removal of the pressure-pin direction it
is symmetric positive definite (SPD), but it is dense. Variable viscosity also
changes the velocity inverse inside the Schur complement, so the entire reduced
operator must be reconstructed at every time step. The border-only shortcut
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

Velocity Dirichlet values and one pressure value are imposed by
`apply_dirichlet_sym`: known values are lifted to the right-hand side and both
the corresponding rows and columns are zeroed before a unit diagonal is
inserted. Therefore the matrix actually used below is still symmetric. The
Schur blocks are obtained by slicing this already-eliminated KKT matrix; the
boundary algebra is not independently reimplemented.

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
dA      = decomposition(A, 'chol');
Y       = dA \ full(Gt);               % Y = A_n^{-1} G_n^T
Sfull   = full(D) + Gt' * Y;
rhsFull = Gt' * (dA \ b1) - b2;
Sfull   = (Sfull + Sfull') / 2;
```

Thus construction of the complete Schur matrix requires one current Cholesky
factorization of $A_n$ and a block solve with all $n_P+n_C$ columns of
$G_n^\mathsf{T}$. The final averaging removes roundoff-level asymmetry; it does
not change the mathematical operator.

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
pressure and multiplier coordinate, and `S_n` is formed as a full MATLAB
matrix. An incomplete sparse factorization is not a natural preconditioner for
this explicit operator; the exact-factor baseline is dense Cholesky.

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
  multiplier-multiplier blocks can all change. The full dense $S_n$ and reduced
  right-hand side are rebuilt.

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
| `deflate_exact` | exact 20-dimensional smallest-eigenvector basis of `S_1`, reused in physical coordinates |
| `deflate_gaussian` | Gaussian inverse-power sketch of `S_1^{-1}` with requested width 20, reused in physical coordinates |

Three objects have distinct refresh rules:

1. The Cholesky of $A_n$ used to **construct** the exact current $S_n$ is
   rebuilt every step because viscosity changes $A_n$.
2. The `chol` solver arm deliberately freezes the dense Cholesky of $S_1$ and
   reuses it as a preconditioner for later $S_n$. This is the factor that
   becomes stale in moving-viscosity cases.
3. Each deflation arm freezes its step-1 basis $V$, while its small Galerkin
   matrix $V^\mathsf{T}S_nV$ is built from the current Schur matrix.

The deflation preconditioner acts directly on the current SPD matrix:

```math
P_n=(I-VV^\mathsf{T})
+\tau V(V^\mathsf{T}S_nV)^{-1}V^\mathsf{T},
\qquad
\tau=\lambda_{\max}(S_1).
```

The basis lives in the physical coordinates of $S_n$. There is no inner split
factor and therefore no coordinate transport. No `ichol` or sparse-proxy arm
is included because the exact Schur complement being studied is dense.

For `deflate_gaussian`, the code applies the requested inverse-power rounds and
then calls `orth(real(Y))` once to construct $V$. The numerical range returned
by `orth` is used verbatim: there is no later `orth_trunc`, SVD cutoff,
rank-revealing QR, column slicing, oversampling reduction, or coordinate
projection. `deflation_P_apply` forms $V^\mathsf{T}S_nV$ from every returned
column.

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
varvisc_schur_extract_examples

cd tests
run_all_tests
```

Smoke mode executes three stress-case steps on an `h0=0.1` mesh without
changing `Tstep`, so it preserves the production motion. Full runs use
`h0=0.05`, 60 solves, and all four cases.

Outputs include `all_results.csv`, `speedup_summary.csv`, per-case solver and
operator-drift plots, cross-case summaries, and `run_config.{mat,json}`.
Generated output directories and example `.mat` files are ignored by git.

`run_varvisc_schur_spectrum` additionally writes the actual ordered eigenvalue
curves to `spectrum/spectrum_raw_vs_prec.png` and
`spectrum/spectrum_raw_snapshots.png`, alongside CSV spectrum summaries and
conditioning plots.

## 8. Verification and code map

The test suite checks the defining Schur properties:

- `test_varvisc_schur_correctness` compares Schur recovery with full `K\b`,
  checks symmetry and Cholesky success, and verifies velocity boundary values;
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
  growth.

The construction is split between three benchmark-local helpers:

- `varvisc_schur_assemble_kkt.m` assembles the parent KKT pair and applies
  symmetric velocity and pressure boundary elimination;
- `varvisc_schur_context_init.m` stores only time-independent mesh and assembly
  data; and
- `varvisc_schur_step_operator.m` slices the KKT blocks, constructs the complete
  dense Schur matrix, removes the pin, and defines full-solution recovery.

All benchmark-local helpers use the `varvisc_schur` prefix to avoid name
collisions with the KKT and constant-viscosity Schur benchmarks.
