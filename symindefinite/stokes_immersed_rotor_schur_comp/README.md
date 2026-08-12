# Schur complement of the immersed-rotor Stokes KKT system

Does the Schur complement of the symmetric **indefinite** KKT sequence studied in
[`../stokes_immersed_rotor/`](../stokes_immersed_rotor/) have the same properties
as the KKT system itself?

It does not — and the ways it differs are the point of this study. Two of them
shape everything below:

- **$S$ is dense**, so no incomplete factorization applies to it. The baseline is
  an exact dense Cholesky built once and recycled (§2).
- **There is no inner preconditioner**, so the deflation basis lives in the
  physical coordinates of $S$ and the sibling's coordinate-drift problem cannot
  arise. Only the operator moves.

---

## 1. The system

The sibling benchmark solves, at each time step of an immersed-rotor Stokes
problem (a simplified MATLAB port of deal.II step-70),

$$
K(t_n) \;=\;
\begin{bmatrix}
A_{\mathrm{vel}} & B^{\!\top} & C(t_n)^{\!\top} \\
B & -\varepsilon L & 0 \\
C(t_n) & 0 & 0
\end{bmatrix},
\qquad
A_{\mathrm{vel}} = \tfrac{1}{\Delta t} M + \nu A ,
\qquad
\varepsilon = \tfrac{h_0^2}{12\nu},
$$

where $p$ enforces $\nabla\!\cdot u = 0$, the multiplier $\lambda$ enforces
$u = g$ on the immersed solid, and $C(t)$ — the barycentric interpolation onto
the moving Lagrange points — is **the only block that changes with time**.
$A_{\mathrm{vel}}$ is SPD and time-constant. In this benchmark the velocity
Dirichlet DOFs are also fixed, so the symmetrically eliminated block
$A=A_{\mathrm{vel,bc}}$ is time-constant as well.

Writing $G = [\,B;\, C(t)\,]$ and $D = \mathrm{blkdiag}(\varepsilon L, 0)$, so
that $K = \begin{bmatrix} A & G^{\top}\\ G & -D\end{bmatrix}$, elimination of the
velocity block gives the $(p,\lambda)$ Schur complement

$$
S(t_n) \;=\; D + G(t_n)\, A_{\mathrm{vel}}^{-1} G(t_n)^{\!\top},
\qquad
S\,y \;=\; G A^{-1} b_1 - b_2 ,
\qquad
u = A^{-1}\!\left(b_1 - G^{\top} y\right).
$$

The sign and right-hand side follow directly from

$$
A u + G^{\top}y=b_1,
\qquad
G u-Dy=b_2:
\qquad
u=A^{-1}(b_1-G^{\top}y)
$$

and hence

$$
\left(D+GA^{-1}G^{\top}\right)y=GA^{-1}b_1-b_2.
$$

**$S$ is SPD.** The sign matters: $-(D + G A^{-1} G^{\top})$ is negative definite
and `pcg` refuses it.

### What actually changes, and what is frozen

Expanding $G_n=[\,B;\,C_n\,]$ makes the update pattern explicit (before removal
of the pressure-pin index):

$$
S_n=
\begin{bmatrix}
\varepsilon L+BA^{-1}B^{\top} & BA^{-1}C_n^{\top}\\
C_nA^{-1}B^{\top} & C_nA^{-1}C_n^{\top}
\end{bmatrix}.
$$

Thus “only $C_n$ changes” does **not** mean that only the lower-right block of
$S_n$ is updated. It changes the full multiplier border: both pressure–multiplier
cross blocks and the multiplier–multiplier block. Only the pressure–pressure
block $\varepsilon L+BA^{-1}B^{\top}$ is constant.

There are three distinct reuse lifecycles in the implementation:

1. `ctx.dA` applies $A^{-1}$. It is built once and reused **exactly**, not as a
   stale approximation, because $A=A_{\mathrm{vel,bc}}$ is constant. It is used
   with the current $C_n$ to assemble the current $S_n$.
2. The `chol` baseline freezes $\operatorname{chol}(S_1)$ and applies that same
   factor as an approximation to $S_n^{-1}$ at every later step. This is the
   inverse that becomes stale; it is not used to assemble $S_n$.
3. The deflation arms freeze the step-1 basis $V_1$, but
   `deflation_P_apply(V, S, ...)` rebuilds the small coarse matrix
   $V_1^{\top}S_nV_1$ from the **current** $S_n$ on every step. The subspace is
   stale while its Galerkin operator is current.

This is therefore not the alternative pattern “freeze an inverse block and
update only a $C$ block.” The underlying KKT model updates only $C_n$; the code
then propagates that change through every affected block of the exact Schur
complement, while separately recycling a frozen $S_1^{-1}$ as a preconditioner.

### Two traps, both verified against source

1. **The pressure pin makes the raw $S$ indefinite by exactly one eigenvalue.**
   `apply_dirichlet_sym.m:32` sets `K(dofs,dofs) = speye`, so after the pin
   $D_{\mathrm{pin},\mathrm{pin}} = -1$; and since $G_{\mathrm{pin},:} = 0$ that
   index is *fully decoupled* with $S_{\mathrm{pin},\mathrm{pin}} = -1$. `chol`
   hard-fails until it is dropped. Nothing is lost — see
   `tests/test_pin_handling.m`, which checks the offending eigenvector really is
   $e_{\mathrm{pin}}$ and that exactly one eigenvalue is negative.
2. **BC elimination is not re-derived.** The KKT pair is assembled
   byte-identically to `src.stokes.solve_stokes_immersed` and then *sliced*, so
   `K\b` is free ground truth and a whole class of BC bugs cannot occur.

### Correctness gates

The algebra is checked independently of the incremental shortcut:

- `tests/test_schur_incremental.m` constructs
  $D+G_nA^{-1}G_n^{\top}$ from scratch at every tested step and compares it with
  the hoisted-block construction. It also verifies that the pressure–pressure
  block is bit-identical across steps and that the update is confined to the
  multiplier border.
- `tests/test_schur_correctness.m` solves the Schur system, scatters the pinned
  pressure value, recovers $u$, and compares the result with the full `K\b`
  solution for all three motions. The observed relative errors are approximately
  $10^{-13}$--$10^{-14}$ on the coarse test problem.
- `tests/test_pin_handling.m` verifies that the unreduced Schur matrix has exactly
  one negative eigenvalue in the decoupled pinned-pressure direction, and that
  deleting that index leaves the same SPD matrix returned by
  `schur_step_operator`.

---

## 2. $S$ is dense, and that decides the whole arm list

$S$ is formed **explicitly**. The $(p,p)$ block is time-constant, so after a
one-time setup each step costs only $n_C \le 44$ backsolves:

```
ONCE     dA   = decomposition(A_bc,'chol');  Y_B = dA \ GtB
         S_pp = D_pp + GtB'*Y_B                        <- CONSTANT
PER STEP Y_C  = dA \ GtC                               <- nC columns only
         S_n  = [S_pp, GtB'*Y_C; (GtB'*Y_C)', D_cc + GtC'*Y_C]
```

Explicit $S$ gives the sketch an exact inverse apply through one dense Cholesky,
and makes every spectral diagnostic exact (`eig`, not `eigs` with a tolerance).

But $S$ contains $A_{\mathrm{vel}}^{-1}$, and the inverse of a sparse SPD matrix
is full. Measured at $n_S = 1001$:

```
S    : issparse = 0,  100% structurally nonzero,  83% of entries > 1e-12 * max
```

**No incomplete factorization applies to this matrix.** `ichol(S)` errors
outright (*"First argument must be sparse"*), and forcing the true pattern
through — `ichol(sparse(S), 'nofill')` — produced 501 nnz/row, i.e. a *complete*
Cholesky under another name.

An earlier version of this study ran `ichol` and AMG arms anyway, on a sparse
BFBt proxy $\hat S = D + G\,\mathrm{diag}(A_{\mathrm{vel}})^{-1}G^{\top}$
(~17 nnz/row). Both have been **removed**: a preconditioner built from a
*different matrix*, needing an escalating diagonal shift to exist at all, is not
a defensible baseline for this operator. For a dense SPD system the honest
preconditioner is a dense Cholesky, and the only real question is how long one
factorization can be reused.

### The four arms

| arm | what it is |
|---|---|
| `pcg_unprec` | unpreconditioned floor |
| `chol` | **baseline** — dense $\mathrm{chol}(S_1)$ built once and recycled for every later step |
| `deflate_exact` | deflation on $S$, $V$ = exact $m$ smallest eigenvectors of $S$ |
| `deflate_gaussian` | deflation on $S$, $V$ = Gaussian sketch of $S^{-1}$, $q$ rounds of plain block power iteration (`build_sketch_V.m`) |

The projector is built on $S$ **directly, first power**:

$$P \;=\; (I - VV^{\top}) \;+\; \tau\, V (V^{\top} S V)^{-1} V^{\top},$$

via `src.precond.deflation_P_apply`. No squaring and no square root — those exist
only to make the coarse matrix definite when the operator is indefinite, which is
the sibling's problem, not this one. There is **no split factor**: $P$ is itself
symmetric positive definite, so it goes to `pcg` as the preconditioner directly,
with no $B = L^{-\top} P L^{-1}$ composition to get wrong. Every captured mode of
$PS$ sits exactly at $\tau$.

Both claims are asserted numerically, not just stated:
`tests/test_sketch_basis.m` T6 checks that $\mathrm{spec}(PS)$ is exactly "$m$
modes at $\tau$, tail untouched", and T7 that $P$ is symmetric positive definite
in its own right.

**$\tau$ is derived, not inherited.** Deflation *relocates* captured modes to
$\tau$ rather than deleting them, so $\tau$ belongs at the top of the retained
spectrum. `params.tau = []` resolves it at step 1 to $\lambda_{\max}(S_1)$
(measured: 0.762 at $h_0 = 0.05$). The old hard-coded 0.5 was tuned against the
`ichol`-split operator and does not transfer.

### Recycling, and why it needs no transport

`DEFLAT_PREC_REFRESH` is huge, so $V$ is built once at step 1 and reused
**verbatim** at every later step. Because no inner factor is involved, $V$ lives
in the **physical coordinates of $S$** — there are no factor coordinates for it to
drift out of. The sibling's transport machinery
(`transport_V`, `ildl_coordinate_map`, `orth_trunc`) is therefore not merely
unused here but *inapplicable*: removing the inner preconditioner removes the
coordinate-drift problem by construction. What remains is the honest part —
$S(t_n)$ moves away from $S_1$, and a basis chosen at step 1 goes stale with it.

Both deflation arms build $V$ from the same exact step-1 inverse of $S$ — the
frozen Cholesky the `chol` arm already needs — and differ **only** in how the
subspace is extracted from it, a full eigensolve versus a Gaussian sketch. That
is the ablation: `deflate_exact` is the quality *reference* for the sketch, not a
different class of method.

---

## 3. Findings

### 3.1 There is a drift story (`tests/test_baseline_drift.m`)

At $h_0 = 0.1$, 12 steps:

| arm | iterations |
|---|---|
| `chol` frozen at step 1 | **1 → 27–30** |
| unpreconditioned | 101–110 |

The frozen exact inverse goes stale immediately — `InvRelDiff` jumps from
$<10^{-10}$ at step 1 to ~0.09–0.15 thereafter — so there is something for
recycling to beat, while remaining far better than no preconditioner at all.

**Staleness here is oscillatory, not cumulative.** The bar sweeps two full
revolutions over $[0, T_{\max}]$, so $S(t)$ moves away from and back toward $S_1$:
`InvRelDiff` measures 0.145 at step 2 and 0.086 at step 12. Any test or claim
phrased as "drift grows monotonically with $n$" is wrong on this geometry, which
is why `test_baseline_drift` T5 asserts that the frozen inverse is *never
accurate after step 1* rather than that its error increases.

Notably `ReldiffF ≈ 0.11–0.24` per step, against **0.006–0.017** for the sphere
reference experiment. The Schur complement moves *much* more per step than a
drifting diffusivity does — the update is low in **rank** but large in
**magnitude**.

### 3.2 Low-rank structure (`run_schur_lowrank.m`)

$S_n - S_m$ is **exactly zero** throughout the $(p,p)$ block, carries **100%** of
its energy in the multiplier border, and has rank **exactly $2n_C$** at every lag
tested — the structural bound is tight, not merely an upper bound. All asserted,
not plotted.

But low rank is not the same as small:

```
||S_20 - S_1||_F / ||S_20||_F = 0.254     numerical rank 32 of 542
```

**5.9% of the directions carry a 25% relative change.** So the update is
maximally confined and still large — which is why a coarse space chosen once at
step 1 does not stay right.

### 3.3 What deflation can buy, and how wide $V$ has to be (`run_schur_spectrum.m`)

This is the number that sizes `sm_eig`. Deflation moves the $k$ smallest modes to
$\tau = \lambda_{\max}$, so the conditioning a width-$k$ coarse space buys is
exactly $\kappa_{\mathrm{defl}}(k) = \lambda_{\max}/\lambda_{k+1}$:

| $h_0$ | $n_S$ | $\kappa(S)$ | $k{=}5$ | $k{=}10$ | $k{=}20$ | $k{=}30$ | $k{=}50$ | $k{=}100$ | $k{=}200$ |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 542 | 3.6e4 | 8.7e2 | 4.6e2 | 3.1e2 | 3.0e2 | 2.8e2 | 2.1e2 | 1.3e2 |
| 0.07 | 1001 | 7.2e4 | 1.7e3 | 9.5e2 | 6.3e2 | 6.0e2 | 6.0e2 | 5.6e2 | 2.9e2 |
| 0.05 | 1959 | 1.9e5 | 4.5e3 | 2.5e3 | 1.6e3 | 1.6e3 | 1.6e3 | 1.5e3 | 1.1e3 |

The curve **knees at $k \approx 20$** — a 116× reduction at $h_0 = 0.05$ — and is
then essentially flat: $k = 100$ buys a further 7% for five times the coarse
space. `sm_eig` is therefore **20**, measured, not inherited.

> The previous version of this section reported "eigenvalues below 1% of
> $\lambda_{\max}$" and concluded there was only **one** low mode. That
> measurement was made on the *`ichol`-preconditioned* operator
> $T_{\mathrm{sym}} = L^{-1}SL^{-\top}$, and it does not describe the operator
> this study now deflates. On the raw $S$ the same threshold catches **83–99%**
> of the spectrum, because $S$ spans five orders of magnitude unpreconditioned —
> it would say "deflate almost everything" at every mesh size. The threshold
> metric was dropped for the $\kappa_{\mathrm{defl}}(k)$ curve above.

The staleness of the baseline is measured on the same run: with
$R = \mathrm{chol}(S_1)$ frozen, $\kappa(R^{-1}S_nR^{-\top})$ goes $1 \to 35 \to
14$ over steps 1/15/30 at $h_0 = 0.05$ — non-monotone, for the revolution reason
in §3.1.

### 3.4 The benchmark (`run_schur_recycle.m`)

> **Not yet measured on the four-arm registry.** The previous table came from a
> ten-arm registry whose two strongest entries (`ichol`, `amg`) ran on a sparse
> proxy and have been removed, and whose deflation arms acted on the split
> operator rather than on $S$. None of those numbers transfer. Run
> `run_schur_recycle` and fill this in.

What the smoke run ($h_0 = 0.1$, `bar_rotating`, 8 steps) shows so far:

| arm | median iterations |
|---|---|
| unpreconditioned | 106.5 |
| `chol` (frozen at step 1) | **27.5** |
| `deflate_exact` ($m{=}30$) | 62.5 |
| `deflate_gaussian` ($m{=}30$) | 79 |

Deflation clearly helps against no preconditioner (~1.7×) and is clearly **beaten
by the recycled exact factorization** (~2.3×). That ordering is the thing the
full run needs to confirm at $h_0 = 0.05$ and on `disk_translating`, and it is
not obviously wrong: a stale exact inverse still carries the whole operator,
while a width-20 coarse space carries 20 directions out of 1959.

### 3.5 The sketch can only find what the spectrum separates

`tests/test_sketch_basis.m` measures the raw-$S$ spectrum at $h_0 = 0.1$
directly:

$$\lambda(S) \;=\; 1.51\mathrm{e}{-5},\; 1.35\mathrm{e}{-4},\; 3.50\mathrm{e}{-4},\;
5.88\mathrm{e}{-4},\; \dots,\; \lambda_{\max} = 0.542,\qquad \kappa = 3.6\mathrm{e}4 .$$

Unlike the split operator — which had one isolated mode and a bulk packed into
$[0.85, 1.0]$ — the raw $S$ has a genuinely **graded** low end. That is why
deflation has something real to remove here.

Inverse power iteration separates modes at rate
$(\lambda_j/\lambda_{j+1})^{2q}$. With $\lambda_1/\lambda_2 = 0.11$ and
$\lambda_2/\lambda_3 = 0.38$, the sketch recovers the first two modes cleanly at
$q = 2$ (per-mode residuals 0.0002 and 0.012) and degrades through the
near-degenerate interior ($\lambda_{20}/\lambda_{21} = 0.969$): measured per-mode
residuals over the first six modes are

```
0.0002   0.0117   0.1111   0.2616   0.2399   0.4067
```

with a whole-block capture residual of 0.67 at $m = 20$. So the sketch is a
**partial** substitute for the eigensolve here, not the near-exact one it was on
the split operator — which is exactly what §3.4's gap between `deflate_exact`
(62.5) and `deflate_gaussian` (79) shows.

One numerical hazard is worth recording: with the plain (non-reorthogonalized)
power iteration the standing convention requires, large $q$ collapses the block.
At $q = 16$ the measured basis width fell from 30 to **1**. The realized width is
therefore recorded per arm as `deflat_dim`, never assumed to equal `sm_eig`.

### 3.6 `disk_static` is degenerate under warm starting

Following the SPD reference experiment, every arm is warm-started from the
previous step's solution. For `disk_static` the coupling block is constant, the
flow reaches steady state around step ~20, and from then on the warm start is
*already* the solution: **every arm reports 0 iterations**. The frozen `chol`
likewise stays at 1 iteration, since $S$ never moves.

That is the correct answer, not a bug — but it makes `disk_static` a degenerate
control here rather than an informative case, and it means this study is **not**
directly comparable to the sibling KKT benchmark on that case (the sibling does
not warm-start MINRES). Use `bar_rotating` and `disk_translating` for
comparisons.

---

## 4. Layout

| file | role |
|---|---|
| `run_schur_recycle.m` | driver: mesh, params, case loop, CSVs, figures, provenance |
| `solve_schur_sequence.m` | the engine — all arms, one `Astat` per case |
| `schur_context_init.m` | time-constant setup; hoists `dA`, `Y_B`, `S_pp` |
| `schur_step_operator.m` | explicit $S(t_n)$, RHS, pin handling, recovery |
| `schur_assemble_kkt.m` | KKT pair, byte-identical to the engine |
| `schur_make_cfg.m` | per-case geometry/BCs, shared with the tests |
| `make_schur_params.m` | defaults + the arm registry |
| `build_sketch_V.m` | Gaussian sketch of $S^{-1}$ → coarse space (local, not `+src`) |
| `add_schur_paths.m`, `assert_local_helpers.m` | path bootstrap + shadowing guard |
| `run_schur_spectrum.m` | exact spectra, mesh sweep, $\kappa_{\mathrm{defl}}(k)$ — **sets `sm_eig`** |
| `run_schur_lowrank.m` | rank/confinement of the step-to-step update |
| `schur_extract_examples.m` | saves $S(t_n)$ + RHS + ground truth at two steps as `.mat` examples (gitignored, **~211 MB each** at $h_0=0.03$) |
| `write_schur_*.m` | CSVs and figures |
| `replot_schur.m` | redraw all figures from the CSV, no re-solving |
| `benchmark_fig_defaults.m`, `save_benchmark_figure.m`, `solver_style_table.m` | figure style (copied from the sibling; the style table keeps 12 colours so that adding arms back cannot silently wrap the palette) |
| `tests/run_all_tests.m` | runs all six assertion scripts (~3 s) |

`define_motion_list.m` is **reused from the sibling folder by path**, not copied.
Since 11 filenames collide with the sibling's and `addpath` prepends by default,
`add_schur_paths` adds the sibling with `'-end'` and this folder with `'-begin'`,
and `assert_local_helpers()` turns any future path-order slip into a hard error
instead of silently mislabelled figures.

### Example operators

`schur_extract_examples.m` writes `schur_example_h0p03_step{01,09}.mat` holding
`S`, `rhs_S`, `y_ref`, `keep` and `meta`. Two things a consumer needs to know:

- **`S` has the pin index removed** (§1, trap 1). `keep` is the
  $n_S^{\mathrm{full}}\times 1$ logical that maps back:
  `y = zeros(meta.nS_full,1); y(keep) = y_ref; y(meta.pin_node) = meta.pin_val;`
  The `st.recover` handle is deliberately *not* saved — it captures a
  `decomposition` object and a dense $n_U\times n_S$ block.
- **The twin in `../stokes_immersed_rotor/`** (`extract_kkt_examples.m`) saves the
  $\mathcal K(t_n)$ these are the Schur complement of, at the same $h_0$ and the
  same steps. Both `meta` structs carry `normK_fro`, `nnzK`, `norm_b`,
  `normC_fro`, computed by two *independent* assembly implementations
  (`schur_assemble_kkt` here, `build_stokes_sequence` there) marched
  independently — they agree to the last digit, which cross-validates both paths.

Steps 1 and 9 are chosen, not 1 and 30: the bar's point set is $\pi$-symmetric,
so $C(\theta+\pi)=P\,C(\theta)$ exactly and steps 16/31 are near-permutations of
step 1. See the $\theta$ table in `extract_kkt_examples.m`.

## 5. Running

```matlab
cd symindefinite/stokes_immersed_rotor_schur_comp
SMOKE_TEST = 1; run_schur_recycle      % 3 steps, one case
run_schur_spectrum                     % run BEFORE trusting sm_eig
run_schur_recycle                      % 3 cases x 60 steps -> schur_recycle/
run_schur_lowrank
schur_extract_examples                 % S(t_1), S(t_9) -> .mat  (~211 MB each)

cd tests                               % all assertion-style scripts
test_schur_correctness    % the gate: Schur solve + recovery == K\b
test_pin_handling
test_schur_incremental
test_sketch_basis         % the sketch AND the deflation projector on S
test_baseline_drift
test_registry_smoke
```

`SMOKE_TEST` trims `params.max_steps`, **not** `params.Tstep` — `Tstep` sets
`Tmax` and hence the rotor's angular velocity, so shrinking it would change the
geometry under test rather than just shortening the run.

Outputs land in `schur_recycle/` (gitignored).
