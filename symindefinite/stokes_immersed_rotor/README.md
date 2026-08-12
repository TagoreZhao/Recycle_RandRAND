# Stokes immersed rotor — a MATLAB step-70

A simplified MATLAB port of [deal.II step-70](https://dealii.org/current/doxygen/deal.II/step_70.html):
incompressible **Stokes flow** in a channel containing a **moving rigid solid**
that is *immersed* in the fluid mesh (a fictitious-domain / non-matching-grid
method) and coupled to the flow by **distributed Lagrange multipliers**. Each
implicit time step produces a **symmetric indefinite saddle-point system** whose
coupling block changes because the solid moves — i.e. *a sequence of symmetric
indefinite linear systems from an implicit solver*.

> This benchmark is intentionally different from the other six: its per-step
> matrix is **indefinite**, so the SPD preconditioner zoo (ICHOL / AMG /
> deflation / PCG used by `solve_deflate_M_P`) does **not** apply. The solver
> comparison here is **backslash vs MINRES (unpreconditioned) vs MINRES with an
> SPD block-diagonal preconditioner**.

## 1. Physical meaning

$u$ is the fluid velocity, $p$ the pressure. A rigid body is held inside the
flow and forced to move on a prescribed path: a **rotating bar** (a stirrer /
rotor), a **translating disk** (a particle advecting down a channel), or a
**static disk** (a fixed obstacle, used as a no-motion baseline). The fluid is
forced to match the body's rigid velocity wherever the body sits; the Lagrange
multiplier $\lambda$ is the distributed reaction force the body exerts on the
fluid. As the body moves, the set of fluid degrees of freedom it "grabs"
changes, so the coupling block $C(t_n)$ — and hence the linear system —
rearranges every step. This is the immersed-boundary analogue of the
moving-`kappa` mechanism in the other benchmarks (cf. the geometry-pairing table
in `.claude/skills/new-benchmark-workflow/SKILL.md`: `rect_with_hole` ↔ flow
past a cylindrical obstacle, wake / vortex shedding).

## 2. Governing equations (continuous problem)

Incompressible **unsteady Stokes** flow on the channel
$\Omega = [0,4]\times[0,1]$, containing a rigid solid that occupies the moving
region $B(t)\subset\Omega$:

```math
\partial_t u \;-\; \nu\,\Delta u \;+\; \nabla p \;=\; f
\qquad\text{in } \Omega,
```

```math
\nabla\!\cdot u \;=\; 0
\qquad\text{in } \Omega,
```

```math
u \;=\; g(t)
\qquad\text{on } B(t)\quad(\text{enforced weakly through }\lambda).
```

**Symbols.**

| symbol | meaning |
|---|---|
| $u(x,t)$ | fluid velocity (vector field, two components $u_x,u_y$) |
| $p(x,t)$ | pressure — the Lagrange multiplier that enforces $\nabla\!\cdot u = 0$ |
| $\nu$ | kinematic viscosity |
| $f$ | body force (zero in the benchmark cases; nonzero in the MMS tests) |
| $g(t)$ | prescribed **rigid-body velocity** of the immersed solid on $B(t)$ |
| $\lambda$ | distributed reaction-force density the solid exerts on the fluid |

**What the equation represents.** Dropping the inertial term $u\!\cdot\!\nabla u$
leaves *creeping (Stokes) flow*: viscous and pressure forces balance the body
force instantaneously, which is the correct regime for micro-scale and
highly-viscous flows. The constraint $u = g(t)$ on $B(t)$ says the fluid moves
rigidly with the solid wherever the solid sits — the body *drags* the
surrounding fluid. The price of imposing that constraint is a force, and that
force is exactly the multiplier $\lambda$ (introduced in §3); physically it is
the traction the body applies to the fluid.

**Boundary conditions** (on $\partial\Omega$, distinct from the immersed
constraint on $B(t)$):

- **parabolic inflow** on the left, $u_x = U_\text{in}\,\dfrac{4\,y\,(L_y-y)}{L_y^2}$, $u_y = 0$;
- **no-slip** on the top and bottom walls, $u = 0$;
- **natural ("do-nothing") outflow** on the right;
- pressure **pinned** at one outflow node to remove the constant-pressure null
  space (Stokes determines $p$ only up to an additive constant).

## 3. Weak form with a distributed Lagrange multiplier

Introduce the spaces $V$ for velocity (incorporating the Dirichlet data), $Q$
for pressure, and $\Lambda$ for the multiplier, which **lives only on the solid**
$B(t)$. The variational problem is: find $u\in V$, $p\in Q$, $\lambda\in\Lambda$
such that for all $v\in V$, $q\in Q$, $\mu\in\Lambda$,

```math
\begin{aligned}
(\partial_t u,\,v) \;+\; \nu\,(\nabla u,\,\nabla v) \;-\; (p,\,\nabla\!\cdot v)
\;+\; \langle \lambda,\,v\rangle_{B(t)} &= (f,\,v),\\[2pt]
(\nabla\!\cdot u,\,q) &= 0,\\[2pt]
\langle \mu,\,u\rangle_{B(t)} &= \langle \mu,\,g(t)\rangle_{B(t)}.
\end{aligned}
```

Here $(\cdot,\cdot)$ is the $L^2(\Omega)$ inner product and
$\langle\cdot,\cdot\rangle_{B(t)}$ is the **coupling pairing on the moving
solid**. This is the **fictitious-domain / distributed-Lagrange-multiplier**
formulation of Glowinski et al.: the solid is *not* meshed and does *not* cut a
hole in the fluid mesh. Instead the rigid-motion constraint is sampled at a set
of **Lagrange points** $X_k(t)$ placed on $B(t)$, and the multiplier $\lambda$
enforces $u(X_k(t)) = V_k(t)$ there.

Two distinct constraints are imposed by two distinct multipliers, so the system
is a **doubly-saddle-point** problem:

- $p$ enforces incompressibility $\nabla\!\cdot u = 0$;
- $\lambda$ enforces the rigid-body velocity $u = g(t)$ on $B(t)$.

Both contribute zero diagonal blocks, which is the source of the indefiniteness
seen in §5.

## 4. Spatial discretization (equal-order P1–P1 + stabilization + coupling)

Triangulate $\Omega$ with a fixed mesh $\mathcal{T}_h$ of $N$ nodes
(`build_channel_mesh_pde`). Use **continuous piecewise-linear (P1)** elements for
*both* velocity and pressure (equal order). Let $\{\phi_j\}$ be the P1 nodal
basis. The discrete velocity is $u_h = (u_{x,h}, u_{y,h})$ and the discrete
pressure is $p_h = \sum_j p_j\,\psi_j$ with $\psi_j = \phi_j$. The DOF vector is
ordered $[\,u_x;\,u_y\,]$ (two blocks of $N$), then $p$ (one block of $N$), then
the multipliers $\lambda$.

**Constant fluid blocks** are assembled once in `assemble_stokes_blocks`, with
local contributions from `tri_mass_loc` and `tri_stiff_loc`.

**Scalar mass** $D_{ij} = \int_\Omega \phi_i\,\phi_j$, with local matrix

```math
\frac{|e|}{12}\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix}.
```

The velocity mass is $M = \mathrm{blkdiag}(D,D)$.

**Scalar stiffness** $K_{ij} = \int_\Omega \nabla\phi_i\!\cdot\!\nabla\phi_j$,
with local matrix $\dfrac{bb^\top + cc^\top}{4|e|}$ (the P1 gradients $b,c$ are
constant on each element). The vector Laplacian is $A = \mathrm{blkdiag}(K,K)$.

**Discrete divergence** $B \in \mathbb{R}^{N\times 2N}$ (`assemble_divergence`),

```math
B_{k,(j,c)} \;=\; \int_\Omega \psi_k\,\partial_{x_c}\phi_j ,
\qquad c\in\{x,y\},
```

so that $B\,[\,u_x;u_y\,]$ is the discrete $\nabla\!\cdot u_h$. Because the
velocity gradients are element-constant and $\int_e \psi_k = |e|/3$, the entries
reduce to $b_j/6$ (x-component) and $c_j/6$ (y-component). $B^\top$ is the
discrete pressure gradient.

**Pressure-stabilization Laplacian** $L = K$, scaled by

```math
\varepsilon \;=\; \frac{h^2}{12\,\nu}.
```

Equal-order P1–P1 **violates the inf–sup (LBB) condition**, so the discrete
pressure is unstable on its own. The **Brezzi–Pitkaränta** remedy adds the
weakly-consistent term $-\varepsilon\,(\nabla p_h,\nabla q_h)$ to the continuity
equation, i.e. a $-\varepsilon L$ block in the $(p,p)$ position. This block is
**negative semidefinite**, which is what makes the assembled matrix indefinite
(§5) rather than merely a zero-diagonal saddle.

**Moving coupling block $C(t)$** (assembled every step in `assemble_coupling`) —
this is the *only* time-dependent operator. The immersed constraint
$u_h(X_k(t)) = V_k(t)$ is imposed at each Lagrange point $k$ and for each
velocity component. Because $u_h$ is P1, its value at a point is a barycentric
combination of the three nodal values of the **host triangle** that contains the
point:

```math
u_h\big(X_k(t)\big) \;=\; \sum_{j} w_{kj}(t)\,u_j ,
```

where $w_{kj}(t)$ are the barycentric weights of $X_k(t)$ in its host triangle,
obtained from `pointLocation` (which triangle?) and `cartesianToBarycentric`
(which weights?). Each constraint row therefore has only **3 nonzeros per
component**, so $C(t)$ is sparse. Its right-hand side is the stacked prescribed
velocity $g(t) = [\,V_x;\,V_y\,]$, and its row count is

```math
n_C(t) \;=\; 2\,\times\,(\text{number of Lagrange points currently inside }\Omega).
```

## 5. Time discretization → the per-step KKT system

Discretize the velocity time derivative by **backward Euler**,

```math
\partial_t u \;\approx\; \frac{u^n - u^{n-1}}{\Delta t},
```

and test against the P1 basis. The mass matrix multiplies the difference quotient,
producing a $\dfrac{M}{\Delta t}$ contribution on the diagonal of the velocity
block and a known term $\dfrac{M}{\Delta t}\,u^{n-1}$ on the right-hand side
(this is exactly the `M2/dt` block and the `(M2/dt)*u_prev` vector in the code).
Assembling everything from §3–§4 at time $t_n = n\,\Delta t$ gives the
**symmetric indefinite saddle-point (KKT) system**

```math
\underbrace{\begin{bmatrix}
\dfrac{M}{\Delta t} + \nu A & B^{\top} & C(t_n)^{\top}\\[6pt]
B & -\varepsilon L & 0\\[4pt]
C(t_n) & 0 & 0
\end{bmatrix}}_{\mathcal{K}(t_n)}
\begin{bmatrix} u^n\\ p^n\\ \lambda^n \end{bmatrix}
\;=\;
\begin{bmatrix} \dfrac{M}{\Delta t}\,u^{n-1} + M f^n\\[6pt] 0\\[4pt] g(t_n) \end{bmatrix}.
```

**Matrix properties (with justification).**

- **Symmetric.** The off-diagonal blocks are transpose pairs ($B \leftrightarrow B^\top$,
  $C \leftrightarrow C^\top$) and the diagonal blocks $\tfrac{M}{\Delta t}+\nu A$
  and $-\varepsilon L$ are symmetric. Dirichlet and pressure-pin conditions are
  imposed by `apply_dirichlet_sym`, which zeros the row *and* the column of each
  constrained DOF and lifts the known value to the RHS, so symmetry is preserved.
- **Indefinite.** The velocity block $\tfrac{M}{\Delta t}+\nu A$ is SPD
  (positive eigenvalues), while the $-\varepsilon L$ block contributes negative
  eigenvalues and the zero $(\lambda,\lambda)$ block contributes a singular
  direction. The spectrum therefore straddles zero.

Because $\mathcal{K}(t_n)$ is **symmetric indefinite**, the right Krylov method is
**MINRES**, not CG (CG requires SPD), and the usual SPD preconditioner toolkit
does not apply directly.

## 6. The sequence of linear solves

The implicit time integrator turns the single PDE into a **sequence of linear
systems** — one indefinite KKT solve per step. The driver loops
(`solve_stokes_immersed`):

```math
\text{for } n = 1,\dots,T_\text{step}-1:\quad
\text{solve } \mathcal{K}(t_n)\,x^n = b^n,\quad
\text{then advance } u^{n-1} \leftarrow u^n .
```

**What stays constant** (assembled exactly once, before the loop):

- the fluid blocks $M$, $A$, $B$, $L$ and the combined velocity block
  $\tfrac{M}{\Delta t} + \nu A$;
- the mesh $\mathcal{T}_h$ and its `triangulation` object;
- the preconditioner factors (the `ichol` factor of the velocity block and the
  `chol` factor of the pressure mass) — see §7.

These can be reused unchanged because the mesh never moves: the *fictitious-domain*
approach immerses the solid in a fixed background grid rather than re-meshing
around it.

**What changes every step** — and *why*, traced to the discretization:

- **$C(t_n)$ and $g(t_n)$.** The Lagrange points $X_k(t_n)$ are relocated each
  step, so `pointLocation` returns different host triangles and
  `cartesianToBarycentric` returns different weights. *This relocation of the
  immersed constraint is the precise discretization mechanism that makes the
  matrix time-dependent* — it is the immersed analogue of a moving coefficient.
- **The right-hand side** $\tfrac{M}{\Delta t}\,u^{n-1} + M f^n$, which carries
  the previous state forward (and any time-varying body force).
- **The constraint count $n_C(t_n)$.** As points enter or leave $\Omega$ the
  number of constraint rows changes, so the **matrix size and sparsity pattern
  themselves change** from step to step — most dramatically in `bar_rotating`.

**Diagnostics the solver logs** to quantify the motion:

- per-step coupling change
  $\;\dfrac{\lVert C(t_n) - C(t_{n-1})\rVert_F}{\lVert C(t_{n-1})\rVert_F}\;$
  (zero when the solid is static, large when it sweeps across new triangles);
- constraint residual
  $\;\dfrac{\lVert C u^n - g(t_n)\rVert}{\lVert g(t_n)\rVert}\;$
  (how well the ground-truth solution satisfies the rigid-body constraint, ~$10^{-8}$).

**The three motions** (`define_motion_list`) sit at different points on this scale:

- **`disk_static`** — fixed obstacle, so $C$ is *constant*; the coupling-change
  series is ≈ 0. This isolates the cost of the saddle structure itself from the
  cost of motion.
- **`disk_translating`** — a disk advecting down the channel; moderate per-step
  change as it crosses new triangles.
- **`bar_rotating`** — a bar spinning about the channel centre with rigid-body
  velocity $v = \omega \times r$ (≈ 2 revolutions over $[0,T_\text{max}]$); large
  per-step change. This is the **stress case**.

## 7. Solver and preconditioner

Each step is solved by **backslash** ($\mathcal{K}\backslash b$, sparse LU) for
the ground truth (and to advance the state), and by **one Krylov solve per entry
of the solver registry** so the preconditioners are compared on identical
systems. Every entry is MINRES except `gmres_exact_inv_frozen`, whose
preconditioner is indefinite by construction. The baseline registry is:

1. **MINRES, unpreconditioned**;
2. **MINRES with an SPD block-diagonal preconditioner** $P$ (defined below);

plus the incomplete-LDL, frozen-exact-LDL, low-rank GMRES and two-level
deflation families described next.

The preconditioner is

```math
P \;=\; \mathrm{blkdiag}\!\Big(\mathrm{ichol}\big(\tfrac{M}{\Delta t}+\nu A\big),\;\; \tfrac{1}{\nu}D,\;\; I_\lambda\Big),
```

the classic Stokes block preconditioner (velocity block + scaled pressure mass;
see Elman–Silvester–Wathen) extended with an identity block on the multiplier.

Because the two nontrivial blocks of $P$ are built from the **time-constant**
fluid operators, their factorizations are computed **once** and reused for every
step; only the well-conditioned identity multiplier block "sees" the motion. The
expected payoff is **near step-independent iteration counts** for preconditioned
MINRES, in contrast to the unpreconditioned curve that degrades as the solid
moves into the shear layer. The gap between the two MINRES curves is the headline
result, and the reuse of the factorizations across the moving sequence is exactly
the recycling story this benchmark is built to expose.

### Incomplete-LDL and two-level deflation

Two stronger families are also registered, the indefinite-system analogs of the
SPD `report/solve_deflate_M_P` scheme (ICHOL→ILDL, PCG→MINRES):

- **Incomplete-LDL (`ildl_nofill`)** — the indefinite analog of `ichol('nofill')`
  (`make_ildl_precond`): an incomplete $LDL^\top$ of $\mathcal K$ with $|D|$
  formed per 1×1/2×2 block so the smoother $M=CC^\top$ is SPD (a legal MINRES
  preconditioner). It is run as a **split solve**: MINRES on
  $\hat A = C^{-1}\mathcal K C^{-\top}$, recovering $x=C^{-\top}y$.

- **Exact LDL, frozen (`exact_ldl_frozen`)** — the same split solve with the
  smoother's *approximation* removed. `make_ildl_precond(..., 'exact')` drops
  nothing, so $C=S^{-1}P^\top L|D|^{1/2}$ satisfies $CC^\top=|\mathcal K|$ exactly
  and $\hat A = C^{-1}\mathcal K C^{-\top} = \mathrm{sign}(D)$, whose spectrum is
  exactly $\{\pm 1\}$: **2 MINRES iterations** on the matrix it was built from.
  Built at step 1 and then frozen (`EXACT_PREC_REFRESH = Inf`), so its per-step
  count is a direct, smoother-free measurement of how fast an *exact* factor stops
  preconditioning as $C(t_n)$ drifts. Because
  $\mathcal K_n = \mathcal K_1 + \Delta C\,\Sigma^\top + \Sigma\,\Delta C^\top$ is a
  symmetric update of rank $\le 2n_C$, $\hat A_n$ is $\mathrm{sign}(D_1)$ plus a
  rank-$\le 2n_C$ perturbation — this curve is the **floor** the deflation and
  Krylov-recycling arms are trying to reach cheaply, not a competitor to them. On
  `disk_static` ($\mathcal K$ constant) it stays at 2 forever, which is the control
  that proves the freeze itself is sound (`test_exact_ldl_frozen`).

- **GMRES on the exact frozen inverse (`gmres_exact_inv_frozen`)** — the only
  non-MINRES arm, and it cannot be MINRES. `exact_ldl_frozen` above has to
  *SPD-ify* the frozen factor ($M=|\mathcal K_1|$) because MINRES demands an SPD
  preconditioner, and what MINRES then sees is $\mathrm{sign}(D_1)$ plus the
  low-rank update. GMRES carries no such constraint, so this arm uses
  $\mathcal K_1^{-1}$ **signed and verbatim**. Left-preconditioning gives

  ```math
  \mathcal K_1^{-1}\mathcal K_n \;=\; I \;+\; \mathcal K_1^{-1}\big(\mathcal K_n-\mathcal K_1\big),
  ```

  an **identity plus a rank-$r$ update** with $r = 2\,\mathrm{rank}(\Delta C)\le 2n_C$.
  Its minimal polynomial has degree $\le r+1$, so **unrestarted** GMRES must
  terminate in at most $2n_C+1$ iterations — 41 for `bar_rotating` ($n_C=20$),
  $\approx 89$ for the disks, and exactly **1** for `disk_static`, where
  $\Delta C\equiv 0$ makes the preconditioned operator the identity. This is a
  theorem rather than a tuning knob: `lowrank_bound.png` plots the arm against
  $2n_C(t_n)+1$ per case and states on the figure whether the claim held. Counts
  *below* the line mean $\Delta C$ was rank deficient at that step, not that the
  bound is wrong. The restart must stay off (`[]`) — restarting discards the
  Krylov space the argument rests on — and `GMRES_MAXIT` (default 300) must stay
  above $2n_C+1$ or the arm reports its budget instead of the claim. Verified by
  `tests/test_gmres_lowrank.m`, which pins the bound on a synthetic KKT sequence
  of known update rank, on the extracted operator pair, and pins the scalar
  iteration-count contract the engine requires (MATLAB's `gmres` returns a 1×2
  `[outer inner]`). Measured against `exact_ldl_frozen` on the *same* frozen
  factor, the gap is the price of MINRES's SPD requirement.

- **Two-level deflation (`two_level_*`)** — the standard split form
  $B=L^{-\top}PL^{-1}$ ($L=C$): MINRES on $\hat A$ with the indefinite deflation
  projector $P_{\rm def}=(I-\hat V\hat V^\top)+\tau\,\hat V|\hat E|^{-1}\hat V^\top$
  ($\hat E=\hat V^\top\hat A\hat V$, SPD-ified via $|\hat E|^{-1}$) as the inner
  preconditioner. The coarse basis $\hat V$ (the near-zero-$|\lambda|$ modes of
  $\hat A$ that stall MINRES) is built by one of four **V operations**, one solver
  entry each (`build_deflation_V`):
  - `exact` — generalized eig $\mathrm{eigs}(\mathcal K,M,\text{'smallestabs'})$;
  - `gaussian` / `sjlt` — random sketch + power iteration on the exact inverse
    $\hat A^{-1}=C^\top\mathcal K^{-1}C$ (one `decomposition(K)`);
  - `polynomial` — matrix-free Chebyshev high-pass on the **squared** operator
    $\hat A^2$ (whose spectrum is $\ge 0$, so the near-zero-$|\lambda|$ cluster maps
    to the low end and is amplified), with a random start. The reject-band edge is
    `lam_cut_frac`·$\max|\lambda(\hat A)|$; set it near $|\lambda_k|/\max|\lambda|$.

- **Low-rank $A^{-1}B$ sketch (`two_level_lowrank_sketch`)** — the same split
  scheme, the same $P^{1/2}$ coarse correction on $\hat A^2$ and the same $\tau$;
  only the source of $\hat V$ differs, so the comparison against `two_level_exact`
  and `two_level_gaussian` isolates exactly that. With $A_1=\mathcal K_1$ factored
  **once** and frozen and $A_2=\mathcal K_n$ the current system, the directions the
  update moves are the range of the **nonsymmetric** $D = A_1^{-1}(A_2-A_1)$, whose
  dominant left singular subspace is taken by randomized power iteration:

  ```math
  Y \;=\; (DD^\top)^q D\,\Omega,\qquad \Omega\in\mathbb R^{n\times k}\ \text{Gaussian},\qquad \hat V \;=\; \mathrm{orth}\big(C_n^\top Y\big).
  ```

  $D$ is never formed: each block application is one sparse $\Delta\mathcal K$
  multiply plus one **batched** backsolve against the frozen factors (raw `ldl`
  factors, not a `decomposition` — that object does not batch a multi-column
  right-hand side, which is the whole cost argument here; see
  `frozen_ldl_apply.m`). Because
  $\mathcal K_n-\mathcal K_1 = U\mathcal B U^\top$ with $U=[\Delta C,\ \Sigma]$ and
  $\mathcal B$ invertible, the target span is known exactly:

  ```math
  \mathrm{range}(D) \;=\; \mathcal K_1^{-1}\,\mathrm{range}(U), \qquad \dim \le 2n_C ,
  ```

  the same space `lowrank_update_basis` computes in one shot — this arm differs in
  that it **truncates to the dominant $k$** of it. Two consequences, both measured:
  $k$ beyond $2n_C$ buys nothing (the pivoted QR returns fewer than $k$ columns and
  `info.rank_drop` says so, which is why the **effective** dimension `info.ncols`,
  not $k$, is the number to quote), and $k$ *below* $2n_C$ is actively harmful — the
  directions the **update** moved most are not the directions the **operator** is
  worst conditioned in. Measured MINRES iteration counts:

  | case | $2n_C$ | smoother alone | $k$ below $2n_C$ | $k\ge 2n_C$ | `two_level_gaussian` |
  |---|---|---|---|---|---|
  | `bar_rotating`, $h_0=0.1$, step 2 | 48 | 385 | 411 ($k=15$) | **273** | 163 (48 cols) |
  | `disk_translating`, $h_0=0.05$, step 2 | 240 | 907 | 896 ($k=100$) | **597** ($k=250$) | 203 (500 cols) |
  | `disk_translating`, $h_0=0.05$, step 3 | 240 | 1931 | 2119 ($k=100$) | **1184** ($k=250$) | 318 (500 cols) |

  So the arm is a real gain over the smoother alone *only* in the $k\ge 2n_C$
  regime, and it does not reach `two_level_gaussian`, which spends 500 columns
  targeting the actual smallest-$|\lambda|$ modes rather than the update. The sketch
  width is $k=\texttt{LOWRANK\_OVERSAMPLE}\cdot\texttt{LOWRANK\_SM\_EIG}$ (default
  $2\times 125=250$, which clears $2n_C\le 240$ for every case at $h_0=0.05$: $n_C$
  is 48 for the bar and 120 for the disks), and $\hat V$ keeps all $k$ columns. **If
  `define_motion_list` changes $n_C$, re-check that the default still clears
  $2n_C$** — `info.rank_drop` in the cached `lowrank_info` entry reports the margin.

  **Cost, in operations rather than seconds:** per step $(2q+1)k$ batched
  backsolves against the frozen factor (fewer once rank truncation shrinks the
  block — 1210 rather than 1250 at $k=250$), $(2q+1)k$ sparse $\Delta\mathcal K$
  matvecs, one $n\times k$ pivoted QR, and **no refactorization after step 1** —
  against `two_level_gaussian`'s $(q+1)\cdot\texttt{DEFLAT\_SM\_EIG}$ column solves
  through a `decomposition` (which does not batch) and a
  $2\times$`DEFLAT_SM_EIG`-wide $\hat E$ build. What is recycled here is the
  **factorization** of $A_1$, not the subspace: $\hat V$ depends on
  $\Delta\mathcal K=\mathcal K_n-\mathcal K_1$ and is rebuilt every step, so this is
  the one deflation entry that does not use `cached_basis`. On `disk_static`
  $\Delta\mathcal K\equiv 0$, there is no space to build, and the arm degrades to
  plain `ildl_nofill` with **identical** counts — the falsification control.
  See `build_lowrank_sketch_V.m`, `frozen_ldl_context.m` and
  `tests/test_lowrank_sketch_V.m`.

These rebuild as the coupling $C(t_n)$ moves, under independent **refresh
cadences** — one knob per preconditioner component, mirroring the report's
`*_PREC_REFRESH` (default `Inf` = build once and **recycle** across the moving
sequence; set to `N` to rebuild every `N` steps):
`BLOCKJAC_PREC_REFRESH`, `ILDL_PREC_REFRESH`, `DEFLAT_PREC_REFRESH` (the basis
$\hat V$), `DINVERSE_PREC_REFRESH` (the exact-inverse factor for sketched V),
`EXACT_PREC_REFRESH` (the frozen exact-LDL factor), `LOWRANK_REF_REFRESH` (the
frozen `ldl` of the reference system $A_1$ the low-rank sketch differences against).

**Coordinates: $\hat V$ is a representation, not a subspace.** MINRES runs on
$\hat A_n = C_n^{-1}\mathcal K_n C_n^{-\top}$ with $\hat y = C_n^\top x$, so a basis
written in split coordinates denotes the *physical* subspace
$C_n^{-\top}\,\mathrm{span}(\hat V)$. The defaults refresh the ILDL every step
(`ILDL_PREC_REFRESH = 1`) but freeze the basis (`DEFLAT_PREC_REFRESH = Inf`), and
`ldl` re-derives the permutation $p$, the scaling $S$ and the 1×1/2×2 pivot
structure from $\mathcal K_n$ — **88–94 % of $p$ changes per step** on the moving
cases. Reusing the same numbers therefore deflates a *different* physical subspace
every step. This is a change of coordinates on the ambient space, which moves
spans; it is not a change of basis within one, which deflation would not even
notice. Left uncorrected it is catastrophic and silent: at benchmark scale a
frozen $\hat V$ costs **266 iterations against 162 for no coarse space at all**,
and `two_level_exact` — the *exact* smallest-$|\lambda|$ modes — collapses
$54\to291$ in a single step on `bar_rotating` while holding at 62 forever on
`disk_static`, where $\mathcal K$ (hence $C$) never changes.

Every basis cached across steps — the coarse space and the recycled Krylov block
alike — is therefore held in **physical** coordinates $U = C^{-\top}\hat V$ and
re-expressed on use as $\hat V_n = \mathrm{orth}(C_n^\top U)$, which preserves the
physical span exactly ($C_n^{-\top}C_n^\top U = U$) for one sparse multiply and one
pivoted QR per step. See `cached_basis` in `define_solver_list.m`,
`transport_V`/`ildl_coordinate_map` in
`../linear_solves/subspace_recycle/kernel/`, and `test_transport_wiring.m`.

What remains after that repair *is* the genuine trade-off this benchmark exists to
measure: the operator itself moves, by a symmetric perturbation of rank $2n_C$ per
step, so a frozen coarse space degrades gradually rather than instantly.

## 8. Industrial applications

Stokes flow with immersed moving solids is the model problem behind
**micro-mixers and lab-on-a-chip stirrers** (the rotating bar is a magnetic
micro-stirrer), **particle-laden creeping flows** (sedimentation, microfluidic
sorting, blood-cell transport in capillaries — the translating disk), and
**fictitious-domain CFD for moving machinery** (pumps, mixers, turbomachinery
rotors) where re-meshing around moving parts each step is avoided by immersing
the solid in a fixed background grid. The non-matching-grid / distributed-
Lagrange-multiplier formulation is exactly the technique deal.II step-70 was
written to demonstrate for large-scale parallel fluid–structure interaction;
this benchmark reproduces its linear-algebra signature (a moving-coupling
symmetric indefinite KKT solved once per step) at a size where preconditioner
behaviour can be studied directly.

## 9. Verification

The numerical method is checked in `convergence_test.m` by the method of
manufactured solutions on the unconstrained solver and by direct algebraic
checks of the assembled KKT matrix:

- **Spatial order** ≈ 2 for velocity in the $L^2$ norm (consistent with P1);
  pressure ≈ 1.5–2.
- **Temporal order** ≈ 1 (backward Euler).
- **Symmetry**: $\lVert \mathcal{K}-\mathcal{K}^\top\rVert_F / \lVert\mathcal{K}\rVert_F$
  at machine precision.
- **Indefiniteness**: the full eigenvalue spectrum has both signs.
- **Constraint satisfaction**: $\lVert Cu-g\rVert/\lVert g\rVert \sim 10^{-8}$.
- **Coupling change**: median per-step change $\geq 0.02$ for `bar_rotating`,
  confirming the stress case actually stresses the coupling.

## 10. Benchmark structure

`run_benchmark.m` is the driver. It builds the channel mesh once, then for each
motion case (`define_motion_list.m`) calls the engine
`+src/+stokes/solve_stokes_immersed.m`, which time-steps the KKT system and, per
step, solves it by backslash (ground truth) and by MINRES for every entry of the
**solver registry** `define_solver_list.m`. Output mirrors
`report/naca0012/benchmark_final` and is written to `benchmark_final/`
(git-ignored, regenerated on run):

```
benchmark_final/
  all_results.csv              one row per (case, time step); <key>_its / <key>_flag
                               columns per solver, plus relres, diffF (coupling
                               change), backslash_relres, constraint_res, nC,
                               solver_err_last (last solver's error vs backslash)
  speedup_summary.csv          per-case max iteration diff & factor vs the
                               unpreconditioned baseline
  paper_summary_table.csv      per-(geometry, case) mean/std iterations + max factor
  run_config.{mat,json}        params, case list, solver keys/labels
  iteration_vs_timestep/<case>.png      all solvers, iterations vs time step
  summary_plots/all_cases_comparison.png
  <case>/
    <key>_solver_iterations.{csv,png}   per-solver series
    all_solvers_comparison.png
    relative_step_to_step_change.png    per-step ||ΔC||_F/||C||_F
    accuracy.png                        error vs backslash + constraint residual
    lowrank_bound.png                   GMRES iterations vs the 2n_C+1 bound
    coefficient_movie/                  (stress case only)
```

A fast end-to-end check runs with `SMOKE_TEST = true; run_benchmark` (single
stress case, 2 steps).

### Redrawing the figures without re-solving

A full run is 3 cases × 60 steps × 10 Krylov solves, so figure changes do not go
through `run_benchmark`. The plotting lives in its own files
(`plot_solver_curves`, `place_solver_legend`, `save_benchmark_figure`,
`write_case_figures`, `write_iteration_vs_timestep`,
`write_all_cases_comparison`, `write_lowrank_bound_figure`) and both drivers call
the same code, so

```matlab
replot_benchmark                              % benchmark_no_krylov_recycle
replot_benchmark('benchmark_krylov_recycle')
replot_benchmark(dir, 'DryRun', true)         % list what would be overwritten
```

regenerates every PNG from `all_results.csv` + `run_config.*` via
`load_benchmark_stats`. It rewrites **figures only** — the CSVs, `run_config.*`
and `coefficient_movie/` are left alone (`'RewriteCsv', true` opts the
per-solver CSVs back in). Live and replotted figures are verified
pixel-identical by `tests/test_plot_helpers.m`, which also covers the legend
layout, the per-solver style table and the CSV round-trip.

`accuracy.png` shows the error-vs-backslash curve only for runs whose CSV has
the `solver_err_last` column; older results directories get the relative
residual instead, and the figure says so. `lowrank_bound.png` is drawn only for
runs that carry the `gmres_exact_inv_frozen` column; results directories written
before that arm existed replot without it.

### Extracting example operators

`extract_kkt_examples.m` saves $\mathcal K(t_n)$, its RHS, and the backslash
ground truth at **two time steps**, for experiments that want the matrix without
running the benchmark:

```matlab
extract_kkt_examples          % -> stokes_kkt_example_h0p03_step{01,09}.mat
                              %    variables: A, b, x_ref, meta
```

The `(A, b, meta)` names match `symindefinite/linear_solves/extract_system.m`,
so consumers of `stokes_kkt_system.mat` load these unchanged. The files are
**gitignored and regenerable** (~1.4 MB each). All assembly is delegated to
`build_stokes_sequence` — the script adds no new copy of the KKT assembly.

Two choices in it are deliberate and non-obvious:

- **$h_0 = 0.03$, not this folder's benchmark default of 0.05.** It matches
  `make_schur_params.m`, so these artifacts and
  `../stokes_immersed_rotor_schur_comp/schur_extract_examples.m`'s are the *same*
  system in two algebraic forms — $S(t_n)$ is the Schur complement of this
  $\mathcal K(t_n)$. Both `meta` structs carry a shared fingerprint
  (`normK_fro`, `nnzK`, `norm_b`, `normC_fro`) so the two independent assembly
  paths can be checked against each other; they agree to the last digit.
- **Steps 1 and 9, not 1 and 30.** The bar's Lagrange-point set is symmetric
  under a $\pi$ rotation, so $C(\theta+\pi) = P\,C(\theta)$ exactly for a
  permutation $P$ — separation in $\theta$ only counts mod $\pi$. With
  $\theta(n) = 11.8^\circ n$, step 9 is $85.6^\circ$ from step 1 (the maximum),
  while steps 16 and 31 are within $3$–$6^\circ$ of a *permutation* of step 1 and
  would be the worst possible partners. The script asserts
  $\lVert C_9-C_1\rVert_F/\lVert C_1\rVert_F > 10^{-3}$ (measured: 1.41).

### Adding a preconditioner

The solver set is the one extensibility seam. Append a struct to
`define_solver_list.m`. An entry provides **either** a `.build` (a 5th-argument
MINRES apply on $\mathcal K$) **or** a `.solve` (a self-contained solve, used by
the split-operator two-level scheme):

```matlab
% 5th-argument preconditioner apply:
solvers{end+1} = struct( ...
    'key',   'my_precond', ...                 % CSV column + file name stem
    'label', 'MINRES (my preconditioner)', ... % plot legend
    'build', @(pc) @(r) my_apply(r, pc));      % [] for an unpreconditioned solve

% self-contained solve (returns [x, flag, relres, iters]):
solvers{end+1} = struct('key','my_solve', 'label','...', 'build',[], ...
    'solve', @(K,b,tol,mit,pc) my_solver(K,b,tol,mit,pc));
```

`pc` carries the reusable ingredients the engine fills:
`pc.Lc`, `pc.Rp`, `pc.Au_bc` (block-Jacobi `ichol` source), `pc.nu`, `pc.nU`,
`pc.nP`; the per-step `pc.nC`, `pc.K` (current KKT matrix) and `pc.step`; and
`pc.cache`, a per-case `containers.Map` for caching/refreshing factorizations.

The `cached(pc, key, refresh, @() build())` helper builds a factor at most once
per step on a `mod(step-1, refresh)==0` cadence (shared keys reused across solver
entries within a step), so a preconditioner can be given its own
`*_PREC_REFRESH` knob (set in `run_benchmark.m`, `Inf` = build once). Add a new
**V-building operation** by adding a `case` to `build_deflation_V.m`.

`cached` takes an optional fifth argument, a predicate on the cached value that
must also hold before it may be reused:
`cached(pc, key, refresh, @() build(), @(v) numel(v.s) == size(K,1))`. It is for
**frozen** factors (`refresh = Inf`), which unlike the every-step ILDL are
*applied* at later steps and can therefore meet a $\mathcal K$ of a different
size — $n = n_U+n_P+n_C$ shrinks if a Lagrange point leaves the fluid mesh. When
the predicate fails, `cached` rebuilds and warns
(`define_solver_list:cacheShapeChanged`) rather than failing inside `applyCinv`;
the warning matters because a forced rebuild silently *un-freezes* the arm. Omit
the argument and the cadence logic is exactly what it was.

Deflation bases use `cached_basis` instead, which stores the **physical** basis
`U = C^{-T} V` on the same cadence and maps it into the current step's split
coordinates on use (see the coordinates note in §7). Its entries carry two step
stamps: `.step` (when `U` was built — drives the refresh cadence) and `.hstep`
(which step's coordinates `.V` is expressed in — a per-step memo, needed because
`two_level_parts` is called once per two-level entry per step *and* a second time
for `gaussian` via `two_level_krylov`).

Nothing else changes: the CSV columns, per-solver plots, comparison plots,
speedup summary and paper table all discover the new solver automatically.

## 11. References

- deal.II **step-70**, *A fluid structure interaction problem on fully
  distributed non-matching grids* —
  https://dealii.org/current/doxygen/deal.II/step_70.html
- Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*
  (block preconditioning of saddle-point Stokes systems; MINRES).
- Brezzi & Pitkaränta, *On the stabilization of finite element approximations
  of the Stokes equations* (the equal-order P1–P1 stabilization used here).
- Glowinski, Pan, Hesla & Joseph, *A distributed Lagrange multiplier /
  fictitious domain method for particulate flows* (the immersed-coupling idea).
