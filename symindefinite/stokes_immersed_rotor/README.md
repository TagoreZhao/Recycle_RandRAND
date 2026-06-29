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

Each step is solved **three ways** so the methods can be compared on identical
systems:

1. **backslash** ($\mathcal{K}\backslash b$, sparse LU) — ground truth, and the
   solution used to advance the state;
2. **MINRES, unpreconditioned**;
3. **MINRES with an SPD block-diagonal preconditioner** $P$ (defined below).

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

## 10. References

- deal.II **step-70**, *A fluid structure interaction problem on fully
  distributed non-matching grids* —
  https://dealii.org/current/doxygen/deal.II/step_70.html
- Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*
  (block preconditioning of saddle-point Stokes systems; MINRES).
- Brezzi & Pitkaränta, *On the stabilization of finite element approximations
  of the Stokes equations* (the equal-order P1–P1 stabilization used here).
- Glowinski, Pan, Hesla & Joseph, *A distributed Lagrange multiplier /
  fictitious domain method for particulate flows* (the immersed-coupling idea).
