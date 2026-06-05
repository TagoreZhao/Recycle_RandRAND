# Sandbox notes — Stokes immersed rotor (simplified deal.II step-70)

Dev-time scratch. The polished, public-facing description is in
`../README.md`.

## Problem

Backward-Euler **unsteady Stokes** in a 2D channel `Ω = [0,4]×[0,1]` with a
**rigid immersed solid** (a spinning bar, or a disk) that does **not** cut a
hole in the mesh — it is enforced on the fictitious (fluid-everywhere) domain
by **distributed Lagrange multipliers** at points that move with the solid.
This is the mechanism of deal.II step-70, reduced to the smallest thing that
still produces *a sequence of symmetric indefinite linear systems from an
implicit solver*.

## Variational form (P1-P1, Brezzi–Pitkaränta stabilized)

Find `(u, p, λ)` such that, for all `(v, q, μ)`,

```
(u_t, v) + ν(∇u,∇v) + (p, ∇·v) + ⟨λ, v(X)⟩ = (f, v)
(∇·u, q) − ε(∇p,∇q)                          = 0
⟨μ, u(X)⟩                                     = ⟨μ, g⟩
```

with `ε = h²/(12 ν)` (BP stabilization, needed because equal-order P1-P1
violates inf-sup). `X = X(t)` are the Lagrange points sampled on the solid and
`g(t)` is the prescribed rigid-body velocity (`ω×r` for the rotor, constant for
the translating disk).

## Discrete system (per step n, time t_n = n·dt)

```
[ M/dt + ν A    Bᵀ        C(t_n)ᵀ ] [u]   [ M/dt·uⁿ⁻¹ + M f ]
[ B            −ε L        0       ] [p] = [ 0                ]
[ C(t_n)        0          0       ] [λ]   [ g(t_n)           ]
```

- **Symmetric**: `M, A, L` symmetric; off-diagonals are transpose pairs. We use
  the sign convention where the momentum pressure term is `+Bᵀp` so the assembled
  matrix `[A Bᵀ; B −εL]` is symmetric (this makes the discrete pressure the
  negative of the physical pressure — irrelevant for a solver benchmark).
- **Indefinite**: the zero `(λ,λ)` block plus the negative `−εL` block give
  eigenvalues of both signs. ⇒ **MINRES**, not PCG; ICHOL/AMG/deflation
  (`solve_deflate_M_P`) do not apply.
- **Changes per step**: only `C(t_n)` and `g(t_n)` move; `M, A, B, L` are
  assembled once.

## Assembly reuse

- `M = blkdiag(msh.D, msh.D)`, `A = blkdiag(K1, K1)` with `K1` the unit P1
  stiffness from `msh.Itrip/Jtrip/Vunit`.
- `B` (divergence) reuses the per-element gradients `b,c` from
  `tri_stiff_loc.m`; element entry `b_j/6` (x) and `c_j/6` (y) — see
  `+src/+stokes/assemble_divergence.m`.
- `L = K1` (pressure-stabilization Laplacian).
- `C(t_n)` interpolates fluid velocity onto the moving solid points via
  `triangulation` / `pointLocation` / `cartesianToBarycentric` — see
  `+src/+stokes/assemble_coupling.m`.

## BCs

Parabolic inflow on the left, no-slip on top/bottom, natural (do-nothing)
outflow on the right; one pressure DOF pinned at the outflow corner to fix the
reference. Velocity Dirichlet eliminated **symmetrically** (`apply_dirichlet_sym`)
to keep MINRES applicable.

## Verification (see ../convergence_test.m)

Order checks use the **method of manufactured solutions** on the *unconstrained*
Stokes solver (the immersed constraint is exact and does not affect orders):

- spatial: steady MMS, velocity L2 order ≈ 2 (P1),
- temporal: transient MMS, backward-Euler order ≈ 1.

Exact field (unit square), divergence-free:
`u = (g·π·sin πx·cos πy, −g·π·cos πx·sin πy)`, `p = g·cos πx·cos πy`, with
forcing `f = u_t − νΔu − ∇p` and the stabilization consistency term `−εL·p*`
moved to the continuity RHS. Plus symmetry / indefiniteness / per-step coupling
change / sparsity diagnostics.

## Departures from the new-benchmark-workflow contract

- **SPD invariant (#7)**: intentionally violated — the system is indefinite
  (the user asked for this explicitly).
- **Preconditioner zoo / PCG**: replaced by backslash + MINRES (unprec + SPD
  block-diagonal precond).
- **Sparsity invariant (#9)**: the vector velocity block + coupling is denser
  than scalar P1; documented, not enforced.

## Reference

deal.II step-70, "A fluid structure interaction problem on fully distributed
non-matching grids":
https://dealii.org/current/doxygen/deal.II/step_70.html
