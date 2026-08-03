# Coordinate drift in preconditioned subspace recycling

Why a recycled deflation basis is destroyed when the ILDL factor is refreshed, why the
same thing barely happens with `ichol`, and why re-charting the cached subspace repairs it.

Every claim below is a statement about **subspaces**, and every one that can be tested
numerically has an experiment in `experiments/` that either confirms it or fails loudly.
`output/verdicts.csv` is the score: **52 PASS, 0 FAIL, 24 REPORT** at the settings quoted
throughout (`experiments/run_all.m`, ~30 s).

---

## Notation, and one standing requirement

| symbol | meaning |
|---|---|
| $A_n$ | step-$n$ system matrix: $\mathcal K_n$, the symmetric indefinite immersed-rotor KKT matrix, or an SPD kernel-ridge matrix |
| $C_n$ | preconditioner factor; $M_n = C_nC_n^{\top}$ is SPD |
| ILDL | $C_n = S_n^{-1}P_n^{\top}L_n\lvert D_n\rvert^{1/2}$ from `[L,D,p,S] = ldl(A,'vector')` |
| ichol | $C_n = L_n$ from `ichol(A,'nofill')` |
| $\hat A_n$ | $C_n^{-1}A_nC_n^{-\top}$, the split operator MINRES runs on; $\hat y = C_n^{\top}x$ |
| $\Phi_n$ | the **chart map** $\mathcal U \mapsto C_n^{\top}\mathcal U$ on the Grassmannian $\mathrm{Gr}(k,N)$ |
| $T_n$ | the **transport** $C_{n+1}^{\top}C_n^{-\top} = \Phi_{n+1}\circ\Phi_n^{-1}$ |
| $\mathcal U_k(A,M)$ | invariant subspace of the pencil $(A,M)$ for the $k$ smallest $\lvert\lambda\rvert$ — the physical deflation target |
| $\hat V,\ U$ | *any* basis of the chart-side / physical-side space; only their spans are ever used |
| $G$ | the coarse correction actually applied, $\;(I-\Pi)+\sqrt{\tau}\,\hat V(\hat V^{\top}\hat A^{2}\hat V)^{-1/2}\hat V^{\top}$ |
| $\tau,\,k,\,n_C$ | coarse weight (0.5); coarse dimension; number of coupling rows |

**Distances.** Only two, both defined through projectors and therefore blind to the choice
of basis:

```math
d_M(\mathcal X,\mathcal Y)=\bigl\lVert \Pi^{M}_{\mathcal X}-\Pi^{M}_{\mathcal Y}\bigr\rVert_{M}=\sin\theta^{M}_{\max},
\qquad
d_F(\mathcal X,\mathcal Y)=\frac{\lVert \Pi_{\mathcal X}-\Pi_{\mathcal Y}\rVert_F}{\sqrt{2k}}=\sqrt{\tfrac1k\textstyle\sum_i \sin^2\theta_i}.
```

$M = I$ gives the Euclidean gap $d$. Both are metrics on $\mathrm{Gr}(k,N)$, so the triangle
inequality of Thm 2.1 is available. $d$ saturates — one lost direction out of $k$ already
gives $1$ — so $d_F$, the root-mean-square principal angle, is reported alongside it
everywhere. `kernel/gap.m`, `kernel/gap_M.m`.

**The standing requirement.** Deflation depends on $\hat V$ only through the projector onto
its span (Thm 1.3), so this analysis does too. Bases appear inside proofs, never in a
statement. Two quantities are deliberately *not* invariant — the factor-dependent bound in
Lemma 2.2 and $\delta_{\mathrm{gauge}}$ in §2.1 — and that non-invariance is the phenomenon
being described, not an oversight.

---

## 1. The chart is a choice of inner product

**Theorem 1.1 (pencil $\leftrightarrow$ split operator).**
$A u=\lambda M u \iff \hat A\,(C^{\top}u)=\lambda\,(C^{\top}u)$.

*Proof.* $\hat A C^{\top}u = C^{-1}Au = \lambda C^{-1}Mu = \lambda C^{-1}CC^{\top}u = \lambda C^{\top}u$. $\square$

So $\Phi_C$ maps pencil invariant subspaces of $(A,M)$ bijectively to invariant subspaces of
$\hat A$, and $\hat A = C^{\top}(M^{-1}A)C^{-\top}$ is similar to $M^{-1}A$.
Measured: $d(\Phi_C\mathcal U_k,\ \hat A\,\Phi_C\mathcal U_k) = 1.2\cdot10^{-13}$ (ILDL),
$1.5\cdot10^{-11}$ (ichol), with eigenvalues preserved to $10^{-15}$ relative.

**Theorem 1.2 (isometry).** $\langle C^{\top}u,C^{\top}v\rangle_2=\langle u,v\rangle_M$, hence
for all $\mathcal X,\mathcal Y\in\mathrm{Gr}(k,N)$

```math
d\bigl(\Phi_C\mathcal X,\ \Phi_C\mathcal Y\bigr)\;=\;d_M(\mathcal X,\mathcal Y).
```

*Proof.* $C^{\top}$ carries the $M$-inner product to the Euclidean one, so it carries
$M$-orthogonal projectors to orthogonal projectors and preserves every principal angle. $\square$

This is the sentence the whole document turns on: **the preconditioner does not relabel
coordinates, it selects the inner product** in which orthogonality, angle, and therefore
deflation are measured. Refreshing $C$ changes the geometry on an unchanged vector space.
Verified by two disjoint code paths — `gap_M` never forms $C$, `gap` never forms $M$ — and
they agree to $3\cdot10^{-16}$ (ILDL) and $1\cdot10^{-16}$ (ichol).

**Theorem 1.3 (the method is a function on the Grassmannian).** $G$, and the spectrum of
$G\hat A$, depend on $\hat V$ only through $\Pi_{\operatorname{span}\hat V}$; in particular
$G(\hat VQ)=G(\hat V)$ for every $Q\in O(k)$.

**Theorem 1.4 (gauge covariance).** The factor is not determined by $M$: $C$ and $CQ$ give the
same $M$ for every $Q\in O(N)$. Under $C\mapsto CQ$ one has $\hat A\mapsto Q^{\top}\hat AQ$ and
$\hat V\mapsto Q^{\top}\hat V$, and the method is unchanged **provided operator and basis are
regauged together**. Freezing $\hat V$ while $C$ is refreshed regauges the operator and not
the basis.

That last sentence is the entire bug, and Thm 1.3 is why it cannot be waved away as "just a
different basis": $O(k)$ acts *inside* the span and is invisible; $O(N)$ acts on the chart and
moves the span. The experiment separates the two exactly (`exp1`, iteration counts):

| | consistent regauge $(C\!\to\!CQ,\ \hat V\!\to\!Q^{\top}\hat V)$ | frozen $\hat V$, same $CQ$ |
|---|---|---|
| ILDL | 54 → **55** | 54 → **159** |
| ichol | 10 → **10** | 10 → **30** |

The identity behind the first column is exact; the single ILDL iteration of difference is
roundoff, since an integer iteration count is a discontinuous function of a residual norm.
Thm 1.3 is therefore tested at the operator level as well —
$\lVert G(\hat V)Z-G(\hat VQ)Z\rVert/\lVert Z\rVert$ at the level of $\epsilon\,\kappa(E)$,
where $E=\hat V^{\top}\hat A^{2}\hat V$ is ill-conditioned by construction because $\hat A$ is
squared.

---

## 2. Separating the $C$ effect from the $A$ effect

Four subspaces, compared physically under the step-$(n{+}1)$ metric — equivalently, by
Thm 1.2, in the step-$(n{+}1)$ chart:

```math
\mathcal F=C_{n+1}^{-\top}\operatorname{span}\hat V_n,\quad
\mathcal U_n=\mathcal U_k(A_n,M_n),\quad
\mathcal U_n'=\mathcal U_k(A_n,M_{n+1}),\quad
\mathcal U_{n+1}=\mathcal U_k(A_{n+1},M_{n+1}).
```

**Theorem 2.1 (three-term decomposition).**

```math
d_{M_{n+1}}(\mathcal F,\mathcal U_{n+1})\;\le\;
\underbrace{d_{M_{n+1}}(\mathcal F,\mathcal U_n)}_{\delta_{\rm chart}\;:\;C_n\ \text{vs}\ C_{n+1}}
+\underbrace{d_{M_{n+1}}(\mathcal U_n,\mathcal U_n')}_{\delta_{\rm prec}\;:\;M_n\ \text{vs}\ M_{n+1}}
+\underbrace{d_{M_{n+1}}(\mathcal U_n',\mathcal U_{n+1})}_{\delta_{\rm op}\;:\;A_n\ \text{vs}\ A_{n+1}},
```

together with the reverse bound $\;\ge\delta_{\rm chart}-\delta_{\rm prec}-\delta_{\rm op}$, so
the split is not vacuous. Both are the triangle inequality for the metric $d_{M_{n+1}}$.

This is the requested separation. $\delta_{\rm chart}$ is the pure factor effect, with the
operator *and* the physical target both held fixed; $\delta_{\rm op}$ is the pure operator
effect, with the metric held fixed; $\delta_{\rm prec}$ is the cross term in which a changed
preconditioner moves the target itself rather than only its coordinates.

**Lemma 2.2 (bounding the chart term).** In the chart,
$\delta_{\rm chart}=d(\operatorname{span}\hat V_n,\ T_n\operatorname{span}\hat V_n)$ and

```math
\delta_{\rm chart}\;\le\;\lVert (I-T_n)\hat V_n\rVert_2\;\le\;\kappa_2(C_n)\,\frac{\lVert C_{n+1}-C_n\rVert_2}{\lVert C_n\rVert_2}.
```

*Proof.* $(I-\Pi_{T\mathcal S})T\hat V=0$, so $(I-\Pi_{T\mathcal S})\hat V=(I-\Pi_{T\mathcal S})(I-T)\hat V$;
take norms. $\square$ The left side is basis-free; the right side is not, and §2.1 turns that
into a theorem.

**Theorem 2.3 (Davis–Kahan for $\delta_{\rm op}$).** $\mathcal U_n'$ and $\mathcal U_{n+1}$ are
pencil subspaces of the *same* metric, so by Thm 1.2 the standard symmetric Davis–Kahan
theorem applies in chart $n{+}1$ unchanged:
$\delta_{\rm op}\le\lVert C_{n+1}^{-1}\,\Delta A\,C_{n+1}^{-\top}\rVert_2/\gamma$ with $\gamma$
the spectral gap of $\hat A_{n+1}$ at index $k$. For the immersed rotor $\Delta A$ has rank
exactly $2n_C$ — measured 32/32 at every step — so Weyl interlacing moves eigenvalues by at
most $2n_C$ places.

![Three-term decomposition](figures/decomposition_bars.png)

*The three terms of Thm 2.1, measured separately, on a shared axis. At one full rotor step
all three ILDL terms are of the same order; the ichol terms are three to four decades
smaller and grow smoothly with the step. The 2-norm gap saturates at $1.000$ for every ILDL
entry, which is why $d_F$ is plotted.* (`exp2`; triangle inequality and reverse bound hold at
every step, both families.)

### 2.1 The chart term splits again: metric and gauge

Write $C_{n+1}=\tilde C_{n+1}Q_n$ with $Q_n\in O(N)$ the Procrustes-optimal alignment to $C_n$,
so that $\tilde C_{n+1}$ is the factor of $M_{n+1}$ sitting in $C_n$'s gauge. With
$\mathcal W = \tilde C_{n+1}^{\top}C_n^{-\top}\operatorname{span}\hat V_n$,

```math
\delta_{\rm chart}\;\le\;\underbrace{d(\operatorname{span}\hat V_n,\ \mathcal W)}_{\delta_{\rm metric}}
+\underbrace{d(\mathcal W,\ Q_n^{\top}\mathcal W)}_{\delta_{\rm gauge}} .
```

**Proposition 2.4 (a perfect preconditioner still breaks a frozen basis).** If
$M_{n+1}=M_n$ *exactly* but $C_{n+1}=C_nQ$ with $Q\ne I$, then $\delta_{\rm metric}=0$ while
$\delta_{\rm chart}=\delta_{\rm gauge}$, which is $\Theta(1)$ for generic $Q$.

This is the sharpest form of the failure, because it holds every competing explanation at
zero. `exp3A` runs it with $\lVert M_2-M_1\rVert/\lVert M_1\rVert = 3.5\cdot10^{-16}$:

| | $\delta_{\rm metric}$ | $\delta_{\rm gauge}$ | its: reference \| frozen \| transported \| no coarse space |
|---|---|---|---|
| ILDL | $1.2\cdot10^{-11}$ | $1.000$ | 54 \| **161** \| 55 \| 146 |
| ichol | $4.3\cdot10^{-15}$ | $1.000$ | 10 \| **29** \| 10 \| 29 |

Nothing about the preconditioner changed. A frozen coarse space nonetheless costs 161
iterations against 146 for **no coarse space at all**, and transport restores the reference
count exactly. The culprit is the factor, not the preconditioner.

![Factor motion versus metric motion](figures/gauge_vs_metric.png)

*Left: the relative motion of the factor and of the metric it defines, per step. Right: the
chart term and its metric/gauge split. Under ILDL both parts saturate; under ichol both are
$O(\lVert\Delta A\rVert)$.*

A prediction worth recording as refuted: the ILDL drift is **not predominantly a regauge**.
A Procrustes alignment removes only about 20 % of the factor motion, and the metric moves by
$O(1)$ as well. Prop 2.4 is decisive because a regauge alone *suffices* to destroy a frozen
basis — not because it is the only thing happening.

---

## 3. Why $\delta_{\rm chart}$ is $\Theta(1)$ for ILDL and $O(\lVert\Delta A\rVert)$ for ichol

**Theorem 3.1 (ichol is real-analytic).** Fix a lower-triangular pattern
$\mathcal S\supseteq\operatorname{diag}$. The IC(0) conditions $(LL^{\top})_{ij}=A_{ij}$ on
$\mathcal S$, $L_{ij}=0$ off it, $L_{ii}>0$ determine $A\mapsto L$, and that map is
real-analytic wherever it exists. Hence
$\lVert\Delta C\rVert\le\gamma_L\lVert\Delta A\rVert+O(\lVert\Delta A\rVert^{2})$ and, by
Lemma 2.2, $\delta_{\rm chart}\to0$ with the step. (Proof in A.1.)

The gauge is pinned by the fixed pattern and the sign convention $L_{ii}>0$, so Prop 2.4's
mechanism has no room to act.

**Theorem 3.2 (ILDL is only piecewise-analytic).** $C_n$ factors through three combinatorial
maps, each locally constant on an open dense set and jumping across a switching set:

1. the AMD ordering, a function of $\operatorname{pattern}(A)$ alone;
2. the MC64-style matching that produces the scaling $S$;
3. the Bunch–Kaufman threshold test $\lvert a_{ii}\rvert\ge\alpha\max_j\lvert a_{ji}\rvert$,
   $\alpha=(1+\sqrt{17})/8$, choosing between $1\times1$ and $2\times2$ pivots.

On each cell $C$ is analytic; across a boundary it jumps by $\Theta(1)$. Therefore
$\lVert\Delta C\rVert/\lVert C\rVert\not\to0$ as $\lVert\Delta A\rVert\to0$, and by the
reverse bound of Thm 2.1 the recycled space is $\Theta(1)$ wrong **however small the time
step is**.

![Continuity sweep](figures/continuity_sweep.png)

*The headline measurement. Interpolate toward the next step, $A(s)=A_1+s(A_2-A_1)$, rebuild
the factor at each $s$, hold the physical subspace fixed, and plot the chart term against
$\lVert\Delta A\rVert/\lVert A\rVert$. ILDL: slope $0.00$, pinned at $\delta_{\rm chart}=1$
across nine decades. ichol: slope $1.00$ (`nofill`) and $1.05$ (`ict`, droptol $10^{-3}$),
tracking the analytic reference.* (`exp4`.)

A value-dependent *drop pattern* is not the same hazard as pivoting: dropping an entry below
the tolerance perturbs $C$ by $O(\text{droptol})$, whereas reordering permutes it. That is
why the `ict` curve still has slope 1.

**Proposition 3.3 (an explicit jump).** Two constructions, at opposite ends of the size range.

*The minimal one.* On the family below, the Bunch–Kaufman rule takes a $1\times1$ pivot
exactly when $\varepsilon\ge\alpha$, so the factor has two different one-sided limits at
$\varepsilon=\alpha$:

```math
A(\varepsilon)=\begin{pmatrix}\varepsilon&1\\ 1&0\end{pmatrix},
\qquad
C_+=\begin{pmatrix}\sqrt{\alpha}&0\\ 1/\sqrt{\alpha}&1/\sqrt{\alpha}\end{pmatrix},
\qquad
C_-=\bigl\lvert A(\alpha)\bigr\rvert^{1/2},
```

both exact. MATLAB's `ldl` flips there (pivot counts $1\,\vert\,0$), matches both closed forms
to $8\cdot10^{-7}$ — the residual is the $10^{-6}$ offset of the probes from $\alpha$ — and an
$O(10^{-6})$ perturbation produces a chart gap of $0.307$.

*The production one.* On the real KKT matrix, apply two perturbations **of the same size**
$\eta$: one that rescales existing entries, and one that places $\eta$ at a structurally zero
position.

| $\eta$ | value-only $\delta_{\rm chart}$ | permutation moved | pattern-change $\delta_{\rm chart}$ | permutation moved |
|---|---|---|---|---|
| $10^{-2}$ | 0.995 | 0.5 % | 1.000 | 92 % |
| $10^{-4}$ | 0.156 | 0 | 1.000 | 92 % |
| $10^{-6}$ | $5.6\cdot10^{-4}$ | 0 | 1.000 | 92 % |
| $10^{-8}$ | $5.6\cdot10^{-6}$ | 0 | 1.000 | 92 % |
| $10^{-10}$ | $5.6\cdot10^{-8}$ | 0 | 1.000 | 92 % |
| $10^{-12}$ | $5.6\cdot10^{-10}$ | 0 | 1.000 | 92 % |

Slope $0.96$ on the left, exactly flat on the right. **A single entry of magnitude
$10^{-12}$ in a new position is enough to destroy the recycled space entirely.** This is the
mechanism that fires at every step of the immersed-rotor problem, where Lagrange points
crossing triangle edges change the coupling pattern; the benchmark measures 88–94 % of the
permutation moving per step. In isolation, one added off-diagonal entry reorders 100 % of an
AMD permutation. (`exp5A`, `exp5B`, `exp5C`.)

**Proposition 3.4 ($\lvert D\rvert^{1/2}$, indefinite only).** For a pivot $d\to0$,
$\lvert d\rvert^{1/2}$ is $\tfrac12$-Hölder, not Lipschitz, and $\kappa_2(C)$ grows like
$\lvert d\rvert^{-1/2}$ — measured log-log slope $-0.5000$ against the predicted $-1/2$ —
which inflates Lemma 2.2's bound by exactly that factor. Structurally absent for SPD ichol,
where $D\equiv I$ is absorbed and $L_{ii}$ is bounded below.

| ingredient | ILDL | ichol | continuous in $A$? |
|---|---|---|---|
| ordering | AMD on $\operatorname{pattern}(A)$ | fixed | **no** / yes |
| scaling $S$ | MC64 matching | none | **no** / — |
| pivot choice | Bunch–Kaufman threshold | none | **no** / — |
| $\lvert D\rvert^{1/2}$ | present | absorbed | Hölder-$\tfrac12$ / — |
| drop rule | pattern of $A$ | pattern of $A$, or droptol | yes / yes (droptol: $O(\text{droptol})$ jumps) |

So: **does ichol drift? Yes — but only through the analytic term, at first order in the
step.** Everything that makes ILDL discontinuous is a decision ichol never makes.

---

## 4. What a wrong subspace costs

The operator MINRES actually sees is $G\hat A$, **not** $G\hat AG$. MATLAB's `minres` takes
its fifth argument as the preconditioner $M$ and applies $M^{-1}$; a function handle there
*is* the apply of $M^{-1}$, and `two_level_split_solve` passes the handle that applies $G$.
The distinction is not cosmetic: on an exactly captured mode $G\hat A$ has eigenvalue
$\sqrt{\tau}\,\operatorname{sign}\lambda$ — the textbook deflation target, and the reason
$\tau=0.5$ is a sensible $O(1)$ choice — whereas $G\hat AG$ would give $\tau/\lambda$, which
blows up on precisely the modes being deflated. (Pinned by `test_kernel` T16–T18; see also
the note in §7.)

**Theorem 4.1 (closed form).** Let $\hat A=\operatorname{diag}(\lambda_1,\lambda_2)$ and let the
coarse space be one-dimensional at angle $\theta$ from the target. With $c=\cos\theta$,
$s=\sin\theta$,

```math
E=\lambda_1^{2}c^{2}+\lambda_2^{2}s^{2},\quad
\beta=\sqrt{\tau/E},\quad
\alpha=\lambda_1c^{2}+\lambda_2s^{2},
```

the two eigenvalues of $G\hat A$ are the roots of

```math
\mu^{2}-\bigl[\lambda_1+\lambda_2+(\beta-1)\alpha\bigr]\mu+\beta\lambda_1\lambda_2=0 .
```

At $\theta=0$ they are $\{\sqrt{\tau}\operatorname{sign}\lambda_1,\ \lambda_2\}$. (Derivation
in A.3.) Verified against the assembled operator to $1.3\cdot10^{-15}$ over the whole sweep.

**Corollary 4.2 (the tolerance on the coarse space).** $E$ stops being $\lambda_1^2$-dominated
once $\tan\theta$ exceeds $\lvert\lambda_1/\lambda_2\rvert$, after which
$\beta\approx\sqrt{\tau}/(\lambda_2 s)$ and the small eigenvalue decays like
$\sqrt{\tau}\lambda_1/(s\lambda_2)$. The usable angle is therefore
$O(\lvert\lambda_1/\lambda_2\rvert)$ — proportional to the very ratio that made deflation
worth doing. Measured half-loss angle $1.81\cdot10^{-2}$ against a predicted
$\arctan\lvert\lambda_1/\lambda_2\rvert = 10^{-2}$.

![Cost versus angle](figures/cost_vs_angle.png)

*Left: the $2\times2$ model. The smallest $\lvert\mu\rvert$ holds at $\sqrt{\tau}$ until
$\theta$ reaches $\arctan\lvert\lambda_1/\lambda_2\rvert$ (dotted), then decays back to the
undeflated $\lvert\lambda_1\rvert$. Right: the real KKT matrix with its own ILDL chart, exact
coarse space rotated by $\theta$. 53 iterations at $\theta=0$, 142 with no coarse space, and
the two curves cross near $\theta\approx0.5$.* (`exp6`.)

**Proposition 4.3 (worse than nothing).** $G$ replaces the identity on
$\operatorname{span}\hat V$ by $\sqrt{\tau}(\hat V^{\top}\hat A^{2}\hat V)^{-1/2}$. When that
span is not near-invariant, the substitution damages directions that were fine instead of
repairing ones that were not, and the coarse space costs more than it saves. Confirmed three
independent ways: in the sweep above (149 against 142 at $\theta\approx0.53$); in the
controlled regauge of Prop 2.4 (161 against 146); and at **every step** of the real sequence
in §5 below (201/156/172/193 frozen, against 174/149/165/179 with no coarse space at all).

The failure mode is therefore *not* hypersensitivity to a small angle — degradation with
$\theta$ is gradual. It is that a refreshed factor drives $\theta$ straight to $\pi/2$.

---

## 5. The repair

Cache the **subspace** $\mathcal U_n=C_n^{-\top}\operatorname{span}\hat V_n$, not the numbers,
and re-chart on use:

```math
\operatorname{span}\hat V_{n+1}\;=\;\Phi_{n+1}\mathcal U_n\;=\;C_{n+1}^{\top}\mathcal U_n .
```

**Theorem 5.1 (transport is exact).** Whenever $C_{n+1}^{\top}$ preserves rank,
$C_{n+1}^{-\top}\operatorname{span}\hat V_{n+1}=\mathcal U_n$ exactly. Hence
$\delta_{\rm chart}\equiv0$, and the residual error in Thm 2.1 is
$\delta_{\rm prec}+\delta_{\rm op}$ alone.

*Proof.* $C_{n+1}^{-\top}C_{n+1}^{\top}=I$ on $\mathcal U_n$. $\square$

Measured round-trip gap: $\le 2.2\cdot10^{-14}$ (ILDL, $\kappa_2(C)$ up to $4.7\cdot10^{3}$)
and $\le1.1\cdot10^{-15}$ (ichol). *Which* basis of $C_{n+1}^{\top}\mathcal U_n$ is computed is
mathematically irrelevant by Thm 1.3; it matters only numerically.

**Proposition 5.2 (the numerical price).** $\Phi_{n+1}$ is not a Euclidean isometry:
$\sigma_{\min}(C_{n+1}^{\top}U)\ge\sigma_{\min}(C_{n+1})$, orthonormality is lost, and roughly
$\log_{10}\kappa_2(C_{n+1})$ digits go with it. Column-pivoted QR with a rank test
(`orth_trunc.m`) is required; unpivoted `qr(Y,0)` is not admissible, and a rank drop is the
diagnostic that $C_{n+1}$ has become ill-conditioned in the sense of Prop 3.4. No rank drops
occurred in any run here.

![What the repair buys](figures/transport.png)

*Per step: frozen numbers, transported subspace, no coarse space, and an oracle that rebuilds
the basis from scratch. Under ILDL the frozen bar is taller than the no-coarse-space bar at
every step — Prop 4.3 — while transport moves below it. Under ichol frozen, transported and
oracle are indistinguishable; only the absence of a coarse space costs anything.* (`exp7`.)

| ILDL, step | frozen | transported | no coarse space | oracle | recovery |
|---|---|---|---|---|---|
| 2 | 201 | 149 | 174 | 60 | 0.37 |
| 3 | 156 | 133 | 149 | 54 | 0.23 |
| 4 | 172 | 141 | 165 | 59 | 0.27 |
| 5 | 193 | 162 | 179 | 61 | 0.23 |

**Transport is necessary but not sufficient, and the honest reason is worth stating.** It
removes $\delta_{\rm chart}$ exactly — the one term that is $\Theta(1)$ at every step size and
that bookkeeping alone *can* remove. It cannot touch $\delta_{\rm prec}$ or $\delta_{\rm op}$,
which are properties of the problem, and at these settings those are not small
($d_F\approx0.4$ each, §2). Recovery of the frozen-to-oracle gap is 23–37 %; the benchmark
sees 0.61 at its own scale, and closing the rest needs the rank-$2n_C$ update studied in
`../linear_solves/subspace_recycle/`.

**A claim this study set out to make and could not.** One would like to argue that the
physical target is nearly metric-independent, so that caching it physically is safe:

**Proposition 5.3.** If $Au=\lambda Mu$ with $\lVert u\rVert_M=1$ then
$\lVert Au\rVert_{M^{-1}}\le\lvert\lambda\rvert$, so every unit vector of $\mathcal U_k(A,M)$
lies in the $\eta$-approximate null space of $A$ alone with
$\eta=\lvert\lambda_k\rvert\lVert M\rVert_2$, and

```math
d\bigl(\mathcal U_k(A,M),\ \mathcal N_k(A)\bigr)\;\le\;\frac{\eta}{\lvert\lambda_{k+1}(A)\rvert}
```

*uniformly in $M$.* (Proof in A.4.)

The bound holds for every metric tested — but it is **weak**: $\eta$ carries a factor
$\lVert M\rVert_2$, and for the ILDL metric the right-hand side evaluates to $2\cdot10^{5}$,
i.e. vacuous. The measurement agrees with the bound's pessimism rather than with the hope:
with $A$ held fixed and only the metric refreshed, the target itself moves by $d_F=0.53$
(ILDL) while the chart moves by $0.80$. Both are real. The case for caching the subspace is
therefore the narrower one made above — not that the target is metric-independent.

| ichol | value |
|---|---|
| chart drift, $A$ fixed, metric refreshed | $d_F = 0.0023$ |
| target drift, same pair | $d_F = 0.0131$ |

---

## 6. Side by side

| term | SPD, ichol | symmetric indefinite, ILDL | governing result |
|---|---|---|---|
| $\delta_{\rm chart}$ | $O(\lVert\Delta A\rVert)$; slope $1.00$ measured | $\Theta(1)$; slope $0.00$, pinned at $1$ | Thm 3.1 vs Thm 3.2 |
| — under a pure regauge | $\Theta(1)$ | $\Theta(1)$ | Prop 2.4 |
| — removable by transport? | yes, exactly | yes, exactly | Thm 5.1 |
| $\delta_{\rm prec}$ | $d_F\le1.0\cdot10^{-2}$ | $d_F\approx0.38\!-\!0.43$ | Prop 5.3 (uniform, weak) |
| $\delta_{\rm op}$ | $d_F\le1.2\cdot10^{-2}$ | $d_F\approx0.38\!-\!0.46$; rank $\Delta A = 2n_C$ | Thm 2.3 |
| cost of freezing | 11 vs 10 its (harmless) | 201 vs 60 its, worse than no coarse space | Prop 4.3 |
| cost after transport | 11 its | 149 its | Thm 5.1 |

The answer to *"how do the interesting subspaces of $\hat A_1$ and $\hat A_2$ differ, and which
part comes from $A$ and which from the factor?"* is the first three rows: under ichol every
part is first order in the step and freezing is nearly free; under ILDL the factor
contributes a term that does not shrink with the step at all, and it alone is larger than
everything the operator does.

---

## 7. Two corrections to the surrounding documentation

1. `stokes_immersed_rotor/README.md` §7 states the coarse correction as
   $(I-\hat V\hat V^{\top})+\tau\hat V\lvert\hat E\rvert^{-1}\hat V^{\top}$ with
   $\hat E=\hat V^{\top}\hat A\hat V$. The code path actually run
   (`deflation_Psqrt_apply` via `two_level_split_solve`) is the square-root form on the
   **squared** operator, $G=(I-\Pi)+\sqrt{\tau}\hat V(\hat V^{\top}\hat A^{2}\hat V)^{-1/2}\hat V^{\top}$.
2. Because MINRES applies the fifth argument as $M^{-1}$, the preconditioned operator is
   $G\hat A$, whose captured eigenvalues are $\pm\sqrt{\tau}$. Reading it as $G\hat AG$ (the
   composition one writes by reflex for a split preconditioner) gives $\tau/\lambda$ and
   predicts, wrongly, that the scheme amplifies exactly the modes it deflates.

---

## 8. Layout and how to run

```
coordinate_drift/
├── README.md                      this document
├── kernel/                        gap, gap_M, pencil_subspace, gauge_split,
│                                  deflated_spectrum, make_case, chart_struct,
│                                  save_figure, vrec, add_paths
├── experiments/                   exp1 .. exp8, run_all
├── tests/test_kernel.m            19 unit checks on the primitives
├── figures/                       committed; embedded above
└── output/                        verdicts.csv + per-experiment csv (gitignored)
```

```matlab
cd tests;        test_kernel        % 19 checks, ~5 s -- run this first
cd ../experiments
run_all                             % ~30 s, writes output/verdicts.csv + figures/
run_all(struct('FULL', true))       % benchmark scale; conclusions unchanged
```

`kernel/make_case.m` is the load-bearing design choice: **one interface, two families**, so
every experiment runs the identical code path for `ildl` (the immersed-rotor KKT sequence
with `make_ildl_precond`) and `ichol` (a sparsified RBF kernel-ridge system from
`GP_train/pumadyn32nm` with a fixed-pattern `ichol`). Only $(A_n,C_n)$ differs between the
two columns of every table above. Fast-mode settings: $n=760$ (`bar_rotating`, $h_0=0.15$)
and $n=600$, $k=50$, $\tau=0.5$, four step pairs. Nothing outside this directory is written
to; `+src/`, the benchmark and `GP_train/` are read-only evidence.

---

## Appendix

**A.1 — Thm 3.1.** Order the entries of $\mathcal S$ column by column. The IC(0) equations
determine them by the recursion
$L_{jj}=\bigl(A_{jj}-\sum_{m<j}L_{jm}^{2}\bigr)^{1/2}$ and
$L_{ij}=\bigl(A_{ij}-\sum_{m<j}L_{im}L_{jm}\bigr)/L_{jj}$ for $(i,j)\in\mathcal S$, every sum
running over previously determined entries. Each step is a composition of $+$, $\times$,
division by $L_{jj}$ and a square root of a strictly positive argument, hence real-analytic
on the open set where every $L_{jj}>0$; the implicit function theorem applies because the
Jacobian in $L$ is triangular with diagonal entries $2L_{jj}$ and $L_{jj}$, all nonzero. Composing gives
$\lVert\Delta L\rVert\le\gamma_L\lVert\Delta A\rVert+O(\lVert\Delta A\rVert^{2})$ with
$\gamma_L=\lVert\partial L/\partial A\rVert$ on the relevant neighbourhood. The `diagcomp`
escalation in `build_ichol_robust` selects $\alpha$ from a finite ladder and is the one SPD
discontinuity; it is observable, since $\alpha$ is returned.

**A.2 — the Procrustes split of §2.1.** $\min_{Q\in O(N)}\lVert C_2-C_1Q\rVert_F$ is attained at
$Q=WZ^{\top}$ where $C_1^{\top}C_2=W\Sigma Z^{\top}$, because
$\lVert C_2-C_1Q\rVert_F^{2}=\lVert C_2\rVert_F^{2}+\lVert C_1\rVert_F^{2}-2\operatorname{tr}(Q^{\top}C_1^{\top}C_2)$
and the trace is maximized there. Setting $\tilde C_2=C_2Q^{\top}$ gives
$\tilde C_2\tilde C_2^{\top}=M_2$ and $T=Q^{\top}\tilde T$ with $\tilde T=\tilde C_2^{\top}C_1^{-\top}$;
the displayed inequality is then the triangle inequality applied to
$\operatorname{span}\hat V_n$, $\mathcal W=\tilde T\operatorname{span}\hat V_n$ and
$T\operatorname{span}\hat V_n=Q^{\top}\mathcal W$.

**A.3 — Thm 4.1.** With $v$ a unit vector, $G=I+(\beta-1)vv^{\top}$ and $\beta=\sqrt{\tau/E}$,
$E=v^{\top}\hat A^{2}v$. Then $\det G=\beta$, so
$\det(G\hat A)=\beta\lambda_1\lambda_2$, and
$\operatorname{tr}(G\hat A)=\operatorname{tr}\hat A+(\beta-1)v^{\top}\hat Av
=\lambda_1+\lambda_2+(\beta-1)\alpha$. A $2\times2$ matrix is determined up to similarity by
its trace and determinant, which gives the quadratic. At $\theta=0$: $E=\lambda_1^{2}$,
$\beta=\sqrt{\tau}/\lvert\lambda_1\rvert$, $\alpha=\lambda_1$, and the quadratic factors as
$(\mu-\lambda_2)(\mu-\sqrt{\tau}\operatorname{sign}\lambda_1)$.

**A.4 — Prop 5.3.** Let $u_1,\dots,u_k$ be $M$-orthonormal pencil eigenvectors with eigenvalues
$\lambda_1,\dots,\lambda_k$ and let $u=\sum c_iu_i$ with $\sum c_i^{2}=1$. Then
$Au=\sum c_i\lambda_iMu_i$ and
$\lVert Au\rVert_{M^{-1}}^{2}=\sum c_i^{2}\lambda_i^{2}\le\lambda_k^{2}$. Converting norms,
$\lVert x\rVert_2\le\lVert M\rVert^{1/2}\lVert x\rVert_{M^{-1}}$ and
$\lVert u\rVert_2\ge\lVert M\rVert^{-1/2}$, so the Euclidean-normalized vector satisfies
$\lVert A\tilde u\rVert_2\le\lvert\lambda_k\rvert\lVert M\rVert=\eta$. Splitting $\tilde u$ in
$A$'s own eigenbasis at index $k$ gives
$\lvert\lambda_{k+1}(A)\rvert\,\lVert\tilde u_{\rm high}\rVert\le\lVert A\tilde u\rVert\le\eta$,
and taking the supremum over the subspace yields the stated gap bound. The bound is uniform
in $M$ and, as §5 records, weak whenever $\lVert M\rVert$ is large.

---

## References

Davis & Kahan (1970); Stewart & Sun, *Matrix Perturbation Theory* (principal angles, the gap
metric on $\mathrm{Gr}(k,N)$); Bunch & Kaufman (1977); Duff & Koster (MC64); Amestoy, Davis &
Duff (AMD); Paige & Saunders (1975, MINRES); Greenbaum, *Iterative Methods* (the two-interval
indefinite bound); Frank & Vuik, and Tang, Nabben & Vuik (deflation and two-level methods);
Elman, Silvester & Wathen (Stokes preconditioning).
