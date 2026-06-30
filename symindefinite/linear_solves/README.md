# `linear_solves` — preconditioner test bed (symmetric indefinite)

A scratch area for prototyping preconditioners for the **symmetric indefinite**
saddle-point (KKT) system solved by the `stokes_immersed_rotor` benchmark with
MINRES, without re-running the full FEM time-stepping pipeline.

## Files

| file | purpose |
|------|---------|
| `extract_system.m`   | Assemble one KKT pair `(A, b)` at a fixed `bar_rotating` snapshot (`h0=0.05`) and save it to `stokes_kkt_system.mat`. |
| `make_ildl_precond.m`| Build an **SPD** preconditioner from an **incomplete LDLᵀ** factorization of `A` (reusable; returns apply handles). |
| `test_ildl_minres.m` | Form the preconditioned system and solve it with MINRES; compare to the unpreconditioned solve; save a convergence plot + droptol-sweep CSV. |
| `plot_eigenspectrum.m`| Plot the smallest-500 and largest-500 \|λ\| (log scale) of `A` vs the ILDL-preconditioned operator `M⁻¹A` → `output/eig_abs_spectrum.png`. |

## How to run

```matlab
extract_system      % writes stokes_kkt_system.mat (once)
test_ildl_minres    % builds the preconditioner and runs MINRES
plot_eigenspectrum  % |lambda| spectrum, raw vs preconditioned (log scale)
```

Outputs land in `output/` (git-ignored). `stokes_kkt_system.mat` is git-ignored
too — it is regenerable via `extract_system.m`.

## Why `|D|` (SPD-ification)

MINRES needs a **symmetric positive definite** preconditioner. An LDLᵀ
factorization of an indefinite `A` has an indefinite block-diagonal `D`
(1×1 and 2×2 blocks). We take the absolute value of each block's eigenvalues to
form

```
M = S⁻¹ Pᵀ L |D| Lᵀ P S⁻¹   (SPD),     C = S⁻¹ Pᵀ L |D|^{1/2},   M = C Cᵀ
```

where `[L,D,p,S] = ldl(A)`. This is the standard SYM-ILDL/MINRES trick.
`make_ildl_precond` returns:

- `P.applyMinv`  — `@(r) M⁻¹ r`, the SPD apply (MINRES 5th argument);
- `P.applyCinv` / `P.applyCtinv` — `C⁻¹` and `C⁻ᵀ`, used by `test_ildl_minres`
  to **form the split/preconditioned system** `C⁻¹ A C⁻ᵀ` and recover
  `x = C⁻ᵀ y`.

"Incomplete" is the cheapest level-0 / **no-fill** variant by default: the exact
`L` is restricted to the sparsity pattern of `A` (the LDLᵀ analog of
`ichol('nofill')`). A `'droptol'` mode is also available.

## Later: register in the benchmark

To use this preconditioner inside `stokes_immersed_rotor`, append a solver entry
to `stokes_immersed_rotor/define_solver_list.m`. The engine calls MINRES with the
5th-argument apply handle, so the build closure is just `P.applyMinv`:

```matlab
solvers{end+1} = struct( ...
    'key',   'ildl_nofill', ...
    'label', 'MINRES (incomplete-LDL, no-fill)', ...
    'build', @(pc) make_ildl_precond(pc.K, struct('mode','nofill')).applyMinv);
```

(this requires the per-step KKT matrix `K` to be made available on the `pc`
context — a small one-line addition in `solve_stokes_immersed.m`.)
