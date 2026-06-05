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

`u` is the fluid velocity, `p` the pressure. A rigid body is held inside the
flow and forced to move on a prescribed path: a **rotating bar** (a stirrer /
rotor), a **translating disk** (a particle advecting down a channel), or a
**static disk** (a fixed obstacle, used as a no-motion baseline). The fluid is
forced to match the body's rigid velocity wherever the body sits; the Lagrange
multiplier `λ` is the distributed reaction force the body exerts on the fluid.
As the body moves, the set of fluid degrees of freedom it "grabs" changes, so
the coupling block `C(t_n)` — and hence the linear system — rearranges every
step. This is the immersed-boundary analogue of the moving-`kappa` mechanism in
the other benchmarks (cf. the geometry-pairing table in
`.claude/skills/new-benchmark-workflow/SKILL.md`: `rect_with_hole` ↔ flow past
a cylindrical obstacle, wake / vortex shedding).

## 2. Governing equations

Incompressible unsteady Stokes on `Ω = [0,4]×[0,1]`, with an immersed solid
region `B(t)`:

```
u_t − ν Δu + ∇p = f      in Ω
∇·u = 0                  in Ω
u = g(t)                 on the immersed solid B(t)   (enforced weakly by λ)
```

- **BCs**: parabolic inflow on the left, no-slip on top/bottom walls, natural
  (do-nothing) outflow on the right; pressure pinned at one outflow node.
- **Discretization**: equal-order **P1-P1** velocity/pressure with
  Brezzi–Pitkaränta pressure stabilization `ε = h²/(12ν)`. The immersed
  constraint is imposed at Lagrange points sampled on the solid via barycentric
  interpolation of the fluid velocity (`triangulation` / `pointLocation`).
- **Time stepping**: backward Euler. Per step the system is

  ```
  [ M/dt + ν A    Bᵀ      C(t_n)ᵀ ] [u]   [ M/dt·uⁿ⁻¹ + M f ]
  [ B            −ε L      0       ] [p] = [ 0                ]
  [ C(t_n)        0        0       ] [λ]   [ g(t_n)           ]
  ```

  which is **symmetric** (transpose-paired off-diagonals) and **indefinite**
  (zero `(λ,λ)` block + negative `−εL`).

- **Expected convergence** (verified in `convergence_test.m` by the method of
  manufactured solutions on the unconstrained solver): velocity spatial order
  ≈ 2 (P1, L2), backward-Euler temporal order ≈ 1. Pressure ≈ 1.5–2.

## 3. Industrial applications

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

## 4. Where this benchmark stresses solvers / preconditioners

The SPD preconditioner suite is **not applicable** (the matrix is indefinite),
which is the point: it marks the boundary of what the rest of the suite can do.
The hypotheses this benchmark tests:

- **Unpreconditioned MINRES degrades as the solid moves into the shear layer**:
  the coupling rows change the spectrum each step, so iteration counts track the
  per-step coupling change `‖ΔC‖_F/‖C‖_F`.
- **The SPD block-diagonal preconditioner** `P = blkdiag(ichol(M/dt+νA),
  M_p/ν, I_λ)` — the classic Stokes block preconditioner extended with an
  identity multiplier block — should give nearly **step-independent** iteration
  counts, because the velocity and pressure-mass blocks are time-constant and
  only the (well-conditioned) multiplier block moves. The gap between the two
  MINRES curves is the headline result.
- **`disk_static` (baseline)** has a constant `C`, so its coupling-change series
  is ≈ 0; contrasting it with `bar_rotating` isolates the cost attributable to
  motion rather than to the saddle structure itself.

## 5. References

- deal.II **step-70**, *A fluid structure interaction problem on fully
  distributed non-matching grids* —
  https://dealii.org/current/doxygen/deal.II/step_70.html
- Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*
  (block preconditioning of saddle-point Stokes systems; MINRES).
- Brezzi & Pitkaränta, *On the stabilization of finite element approximations
  of the Stokes equations* (the equal-order P1-P1 stabilization used here).
- Glowinski, Pan, Hesla & Joseph, *A distributed Lagrange multiplier /
  fictitious domain method for particulate flows* (the immersed-coupling idea).
