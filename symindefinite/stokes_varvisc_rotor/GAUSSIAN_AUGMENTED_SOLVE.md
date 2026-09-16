# Gaussian augmented recycle-range solve

The Gaussian augmented solver reuses a Gaussian range basis across a sequence of symmetric indefinite systems. At each step it adds a fresh Krylov basis from the complement of the recycled space, then uses the combined basis in an SPD coarse correction for MINRES.

This document describes only the `two_level_aug_gaussian` solver, implemented by [varvisc_gaussian_augmented_solve.m](varvisc_gaussian_augmented_solve.m).

## 1. Systems and coordinates

Write the sequence of physical linear systems as

$$
A_i x_i = b_i, \qquad A_i = A_i^T.
$$

The implementation calls the physical matrix `K`. The incomplete-LDL construction provides an SPD preconditioner with split factor $C_i$:

$$
M_i = C_i C_i^T, \qquad
\widehat{A}_i = C_i^{-1} A_i C_i^{-T}, \qquad
\widehat{b}_i = C_i^{-1} b_i.
$$

The split system and solution recovery are

$$
\widehat{A}_i y_i = \widehat{b}_i, \qquad x_i = C_i^{-T} y_i.
$$

Let $V_i$ be an orthonormal basis for the subspace of interest in these split coordinates. At the current step, let $W$ be an orthonormal basis for the projected Krylov subspace defined below; its dependence on $i$ is implicit. All orthogonality and complement statements below use the Euclidean inner product in split coordinates.

## 2. Build and recycle the Gaussian range basis

At the first step, draw a Gaussian matrix $\Omega \in \mathbb{R}^{n \times k}$ and apply the exact inverse of the split operator $q$ times:

$$
V_1 = \mathrm{orth}\!\left(\widehat{A}_1^{-q}\Omega\right),
\qquad
\widehat{A}_1^{-1} = C_1^T A_1^{-1} C_1.
$$

Here $\mathrm{orth}$ denotes an orthonormal basis for the column space. The implementation applies the inverse through a direct factorization of $A_1$ and performs QR after the power rounds. Inverse powers emphasize eigenvectors with small eigenvalue magnitudes, so $V_1$ approximates the corresponding invariant subspace. It is a randomized range approximation, rather than an exact eigenspace.

With the augmented driver's settings, $q=2$ and $k=2\times500=1000$. All sketch columns are retained by the initial QR; there is no reduction back to 500 columns. See [build_deflation_V.m](../../+src/+precond/build_deflation_V.m) and [subspace_iter_plain.m](../../+src/+precond/subspace_iter_plain.m).

The cached basis is stored in physical coordinates:

$$
U = C_1^{-T} V_1.
$$

At step $i$, it is expressed in the current split coordinates and orthonormalized:

$$
\boxed{V_i = \mathrm{orth}(C_i^T U).}
$$

In exact arithmetic, this preserves the physical span:

$$
\mathrm{range}(C_i^{-T}V_i) = \mathrm{range}(U).
$$

By default, the ILDL factor is rebuilt every step (`ILDL_PREC_REFRESH = 1`), while the physical Gaussian basis is frozen (`DEFLAT_PREC_REFRESH = Inf`). Thus $V_i$ is transported from the cached basis, rather than resketched from $A_i$ at every step. The implementation rebuilds the basis if the system dimension changes and can discard numerically dependent columns during transport. The cache and transport logic is in [varvisc_define_solver_list.m](varvisc_define_solver_list.m), in `two_level_parts` and `cached_basis`.

## 3. Construct the fresh projected Krylov space

Define the orthogonal projector onto the complement of $V_i$:

$$
\Pi_i = I - V_i V_i^T.
$$

Use the current split operator and RHS to define

$$
A_{i,\perp} = \Pi_i\widehat{A}_i\Pi_i,
\qquad
r_{i,\perp} = \Pi_i\widehat{b}_i.
$$

For a requested augmentation dimension $m$, the columns of $W$ span

$$
\boxed{
\mathrm{range}(W)
= \mathcal{K}_m(A_{i,\perp},r_{i,\perp})
= \mathrm{span}\!\left\{
r_{i,\perp},\,
A_{i,\perp}r_{i,\perp},\,\ldots,\,
A_{i,\perp}^{m-1}r_{i,\perp}
\right\}.
}
$$

This formula assumes a nonzero starting vector and no early breakdown. The solver supplies no previous solution as an initial guess, so the starting residual is the split RHS before projection.

[varvisc_build_projected_arnoldi.m](varvisc_build_projected_arnoldi.m) constructs this basis without forming a dense projector:

1. Project $\widehat{b}_i$ against $V_i$ twice and normalize the result.
2. For each available vector $w_j$, compute $a=\widehat{A}_i w_j$.
3. Apply two passes of orthogonalization against $V_i$ and the existing columns $w_1,\ldots,w_j$.
4. Normalize the remaining vector to obtain $w_{j+1}$, if another column is needed and breakdown has not occurred.

Consequently, up to rounding error,

$$
V_i^T W = 0, \qquad W^T W = I.
$$

The retained Arnoldi relation is

$$
\widehat{A}_i W = V_i B + W H + f e_{\ell}^T,
\qquad \ell = \mathrm{cols}(W),
$$

where $B$ records the components along $V_i$, $H$ is the projected Arnoldi matrix, $f$ is the final unnormalized remainder, and $e_{\ell}$ is the last coordinate vector. This relation applies when $\ell>0$.

The default is $m=20$. A zero projected start, early breakdown, or the available complement dimension can produce fewer columns; no random padding is added. For $m=0$, $W$ is empty and the solver reduces to Gaussian recycling without augmentation. A short forward Krylov basis is RHS-dependent and need not resolve all eigenvalues nearest zero.

## 4. Form the augmented coarse correction

Append the new directions to the complete recycled basis:

$$
S_i = [\,V_i\; W\,], \qquad S_i^T S_i = I.
$$

The coarse matrix uses the square of the current split operator:

$$
\boxed{
E_i = S_i^T\widehat{A}_i^2S_i
    = (\widehat{A}_iS_i)^T(\widehat{A}_iS_i).
}
$$

For nonsingular $\widehat{A}_i$ and independent columns of $S_i$, $E_i$ is SPD. This is a projection of the squared operator; in general it differs from $(S_i^T\widehat{A}_iS_i)^2$.

The inverse-preconditioner action passed to MINRES is

$$
\boxed{
D_i = (I-S_iS_i^T)
    + \sqrt{\tau}\,S_i E_i^{-1/2}S_i^T,
\qquad \tau=0.5.
}
$$

The symmetric inverse square root is computed from the small eigendecomposition of $E_i$. The code names this action `Pdef`; [deflation_Psqrt_apply.m](../../+src/+precond/deflation_Psqrt_apply.m) constructs it from the squared split operator.

The action is the identity on the complement of $S_i$ and applies a spectral correction within $S_i$. If $S_i$ were an exact invariant subspace of $\widehat{A}_i$, every captured eigenvalue $\lambda$ would be mapped by preconditioning to

$$
\lambda \longmapsto \sqrt{\tau}\,\mathrm{sign}(\lambda).
$$

This explains the intended clustering of captured modes while preserving an SPD preconditioner for the indefinite system. For an approximate invariant subspace, the exact eigenvalue mapping does not apply.

## 5. Run MINRES and advance to the next system

[two_level_split_solve.m](../../+src/+precond/two_level_split_solve.m) runs MINRES on the full split system with $D_i$ as the inverse-preconditioner action. The equivalent symmetric preconditioned system is

$$
D_i^{1/2}\widehat{A}_iD_i^{1/2}z_i
= D_i^{1/2}\widehat{b}_i,
\qquad
y_i = D_i^{1/2}z_i,
\qquad
x_i = C_i^{-T}y_i.
$$

The powers of $D_i$ in this equation describe the mathematical equivalence; the code passes the action of $D_i$ directly to MATLAB's `minres`. The projected operator $A_{i,\perp}$ builds $W$, while the full split operator drives the final solve. The squared operator is used only to construct the coarse correction, so the solver still solves the original indefinite system.

With the default augmented settings, the target coarse dimension is $1000+20=1020$. The actual dimension is $\mathrm{cols}(V_i)+\mathrm{cols}(W)$.

After the solve, $W$ is discarded. At step $i+1$, the solver transports the same cached physical basis $U$ and builds a new $W$ from the current operator and RHS. The Krylov augmentation does not accumulate across timesteps.

MINRES reports convergence in the split/preconditioned solve. The diagnostic path also measures the physical relative residual $\lVert A_i x_i-b_i\rVert_2/\lVert b_i\rVert_2$ separately.

## Enable the Gaussian augmented solver

From the repository root, the existing augmented benchmark driver enables this solver with the settings described above:

```matlab
addpath('symindefinite/stokes_varvisc_rotor');
run_varvisc_augmented_benchmark('full');
```

The driver runs the Gaussian augmented solver alongside the benchmark's comparison solvers. Its augmentation settings are `AUGMENT_ENABLED = true` and `AUGMENT_M = 20`; the ordinary no-argument benchmark does not enable this augmentation by default.
