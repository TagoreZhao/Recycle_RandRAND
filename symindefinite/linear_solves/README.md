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
| `deflation_P_apply_indef.m` | Two-level **deflation** operator `P = (I−VV') + τ·V\|E\|⁻¹V'` for the **indefinite** case (revisable copy of `+src/+precond/deflation_P_apply.m`, with `chol(E)` → `\|E\|⁻¹` SPD-ification so it is a valid MINRES preconditioner). |
| `test_deflation_minres.m` | Validate `deflation_P_apply_indef` on `A` directly: deflation as a standalone SPD MINRES preconditioner, plus an additive `M⁻¹+Q` combination; `k`/`τ` sweep. |
| `test_two_level_minres.m` | **Additive vs multiplicative** two-level head-to-head on the *same* coarse space (eigvecs of the smoothed operator `Â=C⁻¹AC⁻ᵀ`); `k`/`τ` sweep → `output/two_level_minres.{csv,png}`. |
| `test_two_level_sketched.m` | **Exact vs sketched** coarse space of the same size `k`, plus a sweep over the power-iteration count `q` → `output/two_level_sketched.{csv,png}`. |
| `test_two_level_recycle.m` | **Krylov recycling** across two consecutive (perturbed) systems: capture the ILDL-preconditioned residuals of solve 1 for free, append them to the coarse space of solve 2; sweep the recycle count → `output/two_level_recycle.{csv,png}`. |
| `schur_complement/test_two_sided_deflation_pcg.m` | Sketch both spectral tails of the saved SPD Schur system, concatenate and orthogonalize them, then test two sequential deflations with different `tau` values and the same basis using PCG. |

## How to run

```matlab
extract_system        % writes stokes_kkt_system.mat (once)
test_ildl_minres      % builds the preconditioner and runs MINRES
plot_eigenspectrum    % |lambda| spectrum, raw vs preconditioned (log scale)
test_deflation_minres % deflation operator as an SPD MINRES preconditioner
test_two_level_minres % additive vs multiplicative two-level comparison
run('schur_complement/test_two_sided_deflation_pcg.m') % shared-basis, two-stage PCG deflation
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

## Two-level preconditioning: additive vs multiplicative

The ILDL smoother clusters most of the spectrum but leaves a handful of
near-zero eigenvalues (the modes straddling the origin) that stall MINRES. A
**two-level** preconditioner removes those with a small **coarse correction**.
`test_two_level_minres.m` compares the two standard ways of composing the coarse
correction with the smoother, holding the smoother and coarse space fixed.

### Ingredients

- **Smoother** `M = C Cᵀ` (SPD, the `|D|` incomplete-LDL above),
  with `C = S⁻¹Pᵀ L |D|^{1/2}` (so `C⁻¹ = P.applyCinv`, `C⁻ᵀ = P.applyCtinv`).
- **Smoothed operator** `Â = C⁻¹ A C⁻ᵀ` (symmetric; same inertia as `A`, so still
  indefinite). MINRES runs on `Â y = C⁻¹b`, then `x = C⁻ᵀ y`.
- **Coarse space** `V̂ ∈ ℝ^{n×k}`, orthonormal (`V̂ᵀV̂ = I`): the `k`
  smallest-|λ| eigenvectors of `Â`. Computed from the generalized eigenproblem
  `A U = M U Λ` (`eigs(A, M, k, 'smallestabs')`, `U` is `M`-orthonormal), then
  `V̂ = Cᵀ U`.
- **Coarse matrix** `Ê = V̂ᵀ Â V̂ = Λ` (diagonal, **indefinite**). As for the
  smoother, we use the SPD inverse `|Ê|⁻¹ = Ŵ|diag(Ê)|⁻¹Ŵᵀ` and set the SPD
  coarse operator `Q̂ = V̂ |Ê|⁻¹ V̂ᵀ`. (`deflation_P_apply_indef` returns `Q̂` as
  `decE.Qabs` and the multiplicative operator below as its handle `Pdef`.)

### The two compositions

Both run MINRES on the *same* `Â` with an SPD inner preconditioner `G` (the
MINRES 5th argument); equivalently they apply `B = C⁻ᵀ G C⁻¹` to `A`:

```
additive        G_add  = I + Q̂                       B_add  = M⁻¹ + Q
multiplicative  G_mult = (I − V̂V̂ᵀ) + τ·Q̂           B_mult = M⁻¹ − ZZᵀ + τ·Q
```

with `Z = C⁻ᵀV̂` and `Q = Z|Ê|⁻¹Zᵀ` (so `Q = C⁻ᵀ Q̂ C⁻¹`, `M⁻¹ = C⁻ᵀ C⁻¹`). Both
`G` are SPD (on `range(V̂)` they act as `|Ê|⁻¹` resp. `τ|Ê|⁻¹`, on `range(V̂)⊥`
as `I`), hence valid MINRES preconditioners.

### Equivalence and the spectral difference

`B_mult` is **exactly the production scheme** in
`+src/+solver/solve_deflate_M_P.m` (and `report/.../solve_deflate_M_P.m`),
`B = L⁻ᵀ P L⁻¹` with `L ≡ C` and `P = (I−V̂V̂ᵀ)+τV̂Ê⁻¹V̂ᵀ` — the only change is
`chol(Ê)` → `|Ê|⁻¹`, needed because `Ê` is indefinite here.

The two compositions **coincide on `range(V̂)⊥`** (`G·Â = Â` for both) and differ
**only on `range(V̂)`**, where `Â = Λ`:

```
additive        G_add ·Â |_{V̂} = (I + |Λ|⁻¹)Λ = Λ + sign(Λ)   →  λᵢ + sign(λᵢ)   (shift)
multiplicative  G_mult·Â |_{V̂} = (τ|Λ|⁻¹)Λ   = τ·sign(Λ)      →  ± τ            (cluster)
```

i.e. `B_mult − B_add = −ZZᵀ + (τ−1)Q`. Multiplicative *clusters* the deflated
modes onto `±τ`; additive merely *shifts* them away from zero by one. With exact
eigenvectors and `τ = 1` the two are essentially spectrally equivalent (`λᵢ ≈ 0`
⇒ both send the cluster to `≈ ±1`), so they converge in the **same** iteration
count — which is what the test shows. The taxonomy and the AD-vs-DEF/BNN
equivalence theory are from Tang, Nabben, Vuik & Erlangga, *J. Sci. Comput.*
**39** (2009), "Comparison of two-level preconditioners derived from deflation,
domain decomposition and multigrid methods".

### Observed (n≈5840 KKT, no-fill ILDL, exact coarse space)

| composition | k=50 | k=100 | k=250 | k=500 |
|---|---|---|---|---|
| ILDL only | 313 | 313 | 313 | 313 |
| additive `M⁻¹+Q` | 122 | 90 | 62 | 50 |
| multiplicative (τ=1) | 122 | 90 | 62 | 50 |

Both two-level methods give a ~5× iteration reduction at `k=250` and track each
other across `k`; `τ=1` is optimal for the multiplicative form (`τ=0.5` is
slightly worse, `τ=2` ≈ `τ=1`). See `output/two_level_minres_convergence.png`.

### Exact vs sketched coarse space

`test_two_level_sketched.m` repeats the comparison with two coarse spaces of the
same size `k`, side by side:

- **exact** — `V̂ = qr(Cᵀ U)`, `U` from `eigs(A, M, k, 'smallestabs')`
  (eigenvectors of `Â`).
- **sketched** — `V̂ = qr( subspace_iter_plain(Â⁻¹, randn(n,k), q) )`, i.e. a
  Gaussian test matrix pushed through `q` steps of **plain** power iteration on
  the **exact inverse** `Â⁻¹ = Cᵀ A⁻¹ C` (one factorization `decomposition(A)`).
  This reuses `src.precond.subspace_iter_plain` and mirrors the `re==1`
  exact-inverse smallest-eigenvector path (`IcholinvApply`, Gaussian start) in
  `+src/+solver/solve_deflate_M_P.m`. Power iteration on `Â⁻¹` targets the
  smallest-|λ| (near-zero, both-sign) subspace of `Â`.

The composition still does **not** matter much, even with an approximate coarse
space: at every `q` the sketched additive and multiplicative iteration counts
stay within a few iterations of each other (with `τ=1`, `B_mult − B_add = −ZZᵀ`,
a rank-`k` term that barely moves convergence here). The dominant factor is the
**subspace quality**, controlled by `q`:

| q | sketched additive | sketched multiplicative | exact (both) |
|---|---|---|---|
| 0 | 1134 | 1116 | 62 |
| 1 | 462 | 465 | 62 |
| 2 | 214 | 211 | 62 |
| 3 | 162 | 161 | 62 |
| 5 | 105 | 103 | 62 |

(`n≈5840`, `k=250`, `τ=1`; `q=0` is the raw Gaussian sketch.) The sketched coarse
space approaches the exact baseline as `q` grows but does not reach it by `q=5`;
additive and multiplicative track each other throughout. Note: **plain** (non-
reorthogonalized) iteration loses numerical rank as `q` grows — a re-orthogonalized
`subspace_iter` or a sketched-QR stabilization (as in the `+src` `re==2` path)
would be needed to push `q` much higher. See
`output/two_level_sketched_convergence.png`.

### Recycling the Krylov subspace across steps

The benchmark solves a *sequence* of KKT systems that differ only through the
moving coupling block `C(tₙ)`, so the directions MINRES converged slowly on at
step `n−1` are the ones it will converge slowly on at step `n`. Those directions
come for free: MINRES runs on the **split** operator `Â = C⁻¹AC⁻ᵀ`, so the vector
it hands to its preconditioner each iteration *is* the ILDL-preconditioned
residual `C⁻¹(b − Ax)`, the vector that spans the Krylov subspace the solve
explored.

`stokes_immersed_rotor/make_recording_pdef.m` wraps the coarse operator and
records those vectors into a circular buffer keeping the **last** `K` of them.
The capture is a pure side effect of the ordinary `minres` call — no separate
Lanczos, no extra matvec, and the iteration path is bit-identical (asserted in
`test_two_level_recycle.m`). At the next step
`stokes_immersed_rotor/augment_recycle_V.m` forms

```
V = [V_base, orth(W − V_base V_baseᵀ W)],     W = unit-scaled recycled columns
```

so the block is carried **raw** and `orth` is applied only at the point it
becomes a deflation basis — dropping numerically dependent columns there is what
keeps `V̂ᵀÂ²V̂` safely SPD for `deflation_Psqrt_apply`. This is the indefinite/
MINRES port of the SPD/PCG scheme in
`Preconditioner_Recycle/report/ball_surface_krylov_recycle`.

Observed on the frozen `n≈5840` KKT (Gaussian base space `k=100`, `τ=1`, step 2 =
5 % perturbation of the coupling block):

| # recycled | coarse dim | step-2 iters |
|---|---|---|
| 0 | 100 | 127 |
| 10 | 110 | 121 |
| 25 | 125 | 115 |
| 50 | 150 | 113 |
| 100 | 200 | 105 |

Registered in the benchmark as the `two_level_krylov` solver, sharing the cached
`V_gaussian` coarse space with `two_level_gaussian` so the two differ *only* by
the recycled columns (and are identical at step 1, where nothing is recycled
yet). The count is `params.DEFLAT_RECYCLE_K` (default 50).

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
