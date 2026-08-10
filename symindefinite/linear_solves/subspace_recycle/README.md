# Why the recycled deflation subspace fails — and a cheap repair

Diagnosis study for the `stokes_immersed_rotor` benchmark, where two-level
deflation gives ~3–6× at step 1 and **nothing** from step 2 onward, and Krylov
recycling adds ~1%.

Everything here is self-contained; the benchmark and `+src/` are read-only
evidence. Stages 1–3 (kernel, gates, drift/eigenspace diagnostics) are built and
run; `update/` (stage 4) is designed but not yet built.

---

## The short answer

**Two independent failures, in order. The first one has to be fixed before the
second one is even well posed.**

1. **The coarse space is destroyed by a change of coordinates, not by the
   operator moving.** `V` is a *representation*, `V = Cᵀ U`. The benchmark
   refreshes the ILDL factor every step (`ILDL_PREC_REFRESH = 1`) but freezes
   `V` (`DEFLAT_PREC_REFRESH = Inf`). Because the coupling block's **sparsity
   pattern** changes each step (Lagrange points cross triangle edges), `ldl`
   re-pivots: 88–94 % of the permutation changes. The frozen basis then deflates
   a subspace with essentially zero overlap with the eigenvectors it was built
   from — **0 of 100 directions still captured to 1 % at step 2**, capture error
   1.000.

2. **What remains after that is a genuine, and provably small, operator change.**
   With coordinates held fixed, `Âₙ − Â_ref` has rank **exactly 2·nC** (measured
   32/32), eigenvalues move at most 2·nC places in the ordering (interlacing
   verified to 3.5e-16), and the frozen space misses 0.377 of the true
   eigenspace in the Frobenius sense.

**The missing component** is `Ŵₙ = Cₙᵀ(Kₙ⁻¹[dCₙ, Sel])`, at most 2·nC = 40–88
columns. Because `Kₙ − K_ref` maps everything into `range([dCₙ, Sel])`, the span
identity `Kₙ⁻¹·range(U) = K_ref⁻¹·range(U)` holds **exactly** — so a frozen
factorization is not an approximation here, and the cost is `nC ≤ 44` backsolves
per step (`Sel` is time-independent, so `K_ref⁻¹Sel` is computed once).

Together the two repairs recover **84 %** of the gap between today's behaviour
and a full eigensolve at every step.

---

## Results (fast mode: `bar_rotating`, n = 1597, k = 100, τ = 1)

### The failure reproduces exactly

Mean MINRES iterations over steps 2, 3, 5, 10 — `run_drift_factorial`:

| configuration | iterations | vs ILDL |
|---|---|---|
| ILDL only, no coarse space | 193 | 1.00× |
| **production** (frozen V, refreshed C) | **194** | **1.00×** |
| step-1 in-sample (what deflation is worth) | 55 | 3.5× |
| oracle: rebuild V every step | 57 | 3.4× |

A 100-column coarse space — 6 % of the whole system — buys **literally nothing**.
This is the benchmark's signature (312 vs 291 at n = 5840) at quarter scale.

### The two repairs

| repair | per-step cost | iterations | recovery |
|---|---|---|---|
| **transport** `V ← orth(Cₙᵀ U_ref)` (H1) | k spmv + one QR | 114 | 0.61 |
| **rank-2nC block**, coordinates frozen (H2) | nC backsolves, frozen factorization | 120 | 0.50 |
| **both** | both | **82** | **0.84** |
| *control:* transport + the same number of **random** columns | — | 115 | 0.60 |
| oracle (rebuild V) | full `eigs` every step | 57 | 1.00 |

*recovery = (production − cell) / (production − oracle).*

The random-column control is the load-bearing one: **114 → 115 with random
columns, 114 → 82 with the low-rank block.** The block supplies specific missing
directions, not merely a wider space.

### It is the subspace, measured directly

`run_eigenspace_motion`, coordinates frozen so the perturbation is exactly
rank-2nC. Capture of the true step-n smallest-|λ| eigenspace:

| coarse space | +cols | ‖(I−P)V‖₂ | ‖(I−P)V‖_F/√k | dirs captured <1 % |
|---|---|---|---|---|
| frozen `V_ref` | 0 | 0.9999 | 0.377 | 45 / 100 |
| + raw `Û` (no solve) | 32 | 0.935 | 0.283 | 49 |
| **+ rank-2nC block** | 32 | 0.620 | **0.140** | 56 |
| + random columns | 32 | 0.995 | 0.373 | 45 |

2.70× improvement from the block; 1.01× from the same number of random columns.

### The controlled proof that it is coordinates, not the operator

`run_pivot_sensitivity` holds the system **byte-identical** (always `K_ref`) and
only rebuilds the ILDL:

- **Value-only perturbation, pattern preserved:** the `ldl` permutation does not
  move at all up to δ = 0.1 and the frozen basis is intact (55 → 60 iterations).
  It is *not* a hair trigger. At δ = 1 — the magnitude of one real step — 43 % of
  the permutation moves and frozen V degrades 55 → 246, transported → 127.
- **Real pattern change (ILDL built from a later step, operator unchanged):**
  permutation moves 88–94 %, capture falls to 1.000, and at an *identical
  smoother* frozen V is 1.77–1.97× slower than transported V — and no better
  than using no deflation at all (ILDL/frozen = 0.95–0.99).

---

## Three findings that change the plan

1. **Freezing the ILDL is not free.** A mismatched factor is a **2.19× worse
   smoother** (ILDL-only 183 → 402 iterations). "Freeze C to keep the
   coordinates stable" loses more than it gains — the right combination is
   **refresh the factor every step and transport V into it.**

2. **The deflation targets are pressure modes, not constraint modes.** Median
   energy split of the smallest-|λ| eigenvectors: **velocity 0.007, pressure
   0.972, multiplier 0.018.** The Brezzi–Pitkäranta stabilization makes the
   (p,p) block `−ε L` with ε = h²/(12ν), so the near-null space is the stabilized
   pressure field — physical, and spread over ~21 % of the pressure DOFs rather
   than isolated fill spikes. The moving block `C(t)` touches only velocity and
   multiplier rows, so a constraint-driven update reaches these modes only
   indirectly through `C⁻¹`. That is exactly why it recovers 0.50 and not more.

3. **At k = 500 the scheme cannot win on wall clock — but not for the reason a
   flop count suggests.** Measured at n = 5840 (one `Â` apply = 2.6e-4 s, and the
   ILDL-only baseline is 312 iterations = 312 units):

   | k | `E = V'Â²V` rebuild | coarse apply | iterations available to beat ILDL |
   |---|---|---|---|
   | 50 | 105 u | 1.2 u/it | **131** (needs 2.4× — achievable) |
   | 100 | 113 u | 1.8 u/it | 83 (needs 3.8×) |
   | 500 | **546 u** | 4.7 u/it | **impossible: setup alone exceeds the budget** |

   The binding cost is the **per-step `E` rebuild at 2k operator applies**, not
   the per-iteration coarse apply. A naive `8nk` flop count predicts the coarse
   apply at ~68× one operator apply; it is measured at **3.6×**, because the
   operator apply is dominated by sequential sparse triangular solves while the
   coarse apply is threaded dense BLAS-2 — a ~19× discrepancy that
   `matvec_budget` now carries explicitly as `dense_speedup`.

   Practical consequence: **k ≈ 50**, where the repairs' measured 2.4× reduction
   sits right at break-even. It is genuinely marginal, and settling it is
   stage 4's job — not something to claim from a model.

---

## Layout

```
subspace_recycle/
├── kernel/          shared helpers + 7 unit-test files (47 checks)
│   ├── build_stokes_sequence.m   the KKT sequence in low-rank form
│   ├── assert_coupling_feasible.m  refuses an exactly-singular coupling block
│   ├── seq_K.m / seq_dCblk.m     K_n = K0 + Cblk_n Sel' + Sel Cblk_n'
│   ├── ildl_coordinate_map.m     explicit C + pivot/permutation drift
│   ├── transport_V.m             the H1 repair
│   ├── lowrank_update_basis.m    the missing component (raw|invref|exactsolve|shifted)
│   ├── two_level_it.m            instrumented split solve (+ true_res, cond(E))
│   ├── orth_trunc.m              rank-truncating orthonormalization
│   ├── matvec_budget.m           work in matvec-equivalents
│   └── run_kernel_tests.m        runs all seven
└── diagnosis/       + output/ (gitignored)
    ├── run_pivot_sensitivity.m   H1 gate: operator fixed, only the ILDL varies
    ├── run_mode_localization.m   H6 gate: are the targets physical?
    ├── run_ildl_drift.m          per-step drift; is freezing the ILDL free?
    ├── run_drift_factorial.m     separates H1/H2, scores each repair
    └── run_eigenspace_motion.m   what is missed, and the interlacing bound
```

## How to run

```matlab
cd kernel;    run_kernel_tests          % 47 checks, ~5 s — run this first
cd ../diagnosis
run_pivot_sensitivity                   % H1 gate
run_mode_localization                   % H6 gate
run_ildl_drift
run_drift_factorial                     % the headline table
run_eigenspace_motion                   % the capture figure
```

Every script defaults to fast mode (h0 = 0.1, n ≈ 1600, ~2–5 min each). For
benchmark scale (h0 = 0.05, n = 5840, k = 500, 60 steps, 3 cases) set
`FULL = true` in the base workspace first. `run_eigenspace_motion` uses a dense
`eig` in fast mode, which gives the *whole* spectrum and makes the interlacing
check exact; FULL switches to `eigs` and skips it.

Sequences are cached under `kernel/cache/` (gitignored). **Manual deletion is no
longer the mechanism**: the filename tag records only
`(case_name, h0, dt, nsteps)`, so it is blind to `Tstep`/`Tmax`, `nu`, the channel
box, and above all the Lagrange-point layout, which lives inside
`define_motion_list.m` as literals and can never appear in a tag built from
`opts`. Every entry therefore stores a geometry `fingerprint`, and every cache hit
re-derives and compares it; a mismatch — or a legacy entry that has none — warns
and rebuilds. Editing the geometry and re-running is safe. Changing the *assembly*
without changing the geometry still is not: that is what `use_cache = false` is
for.

## Validation

`test_build_stokes_sequence` T7 rebuilds the benchmark's own step-1 system and
reproduces its `ildl_nofill` iteration count **exactly (321)**, so these numbers
describe the benchmark's system and not a lookalike. The low-rank identity holds
to 0.0e+00 for all three motion cases, and `disk_static` is a clean null control
throughout: `dC = 0` exactly, 0 % permutation drift, 100/100 directions captured,
every repair degenerating to the frozen baseline.

## Not yet done (stage 4–5)

`update/` — the variant registry, the full sequence head-to-head, the k-sweep
Pareto front and the `timeit` cost model; plus the Krylov-recycling autopsy and
the τ / coarse-health sweep. The recycling mechanism is already implicated by
these results: it records `W` in step n−1's split coordinates and appends it to a
basis that is itself meaningless in step n's coordinates, so it was augmenting
noise with noise.
