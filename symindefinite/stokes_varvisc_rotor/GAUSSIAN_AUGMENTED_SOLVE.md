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
V_1 = \mathrm{orth}(\widehat{A}_1^{-q}\Omega),
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

Fix the system index $i$. Write $n$ for the current system dimension and $k$ for the number of columns of $V_i$. Define

$$
V_i \in \mathbb{R}^{n\times k}, \qquad
V_i^T V_i = I_k, \qquad
\mathcal{V}_i = \mathrm{range}(V_i).
$$

The equations below hold in exact arithmetic; computed orthogonality and recurrence identities hold up to rounding error. The construction uses $\widehat{A}_i=C_i^{-1}A_iC_i^{-T}$, so the complement is taken in split coordinates.

### 3.1 Projected operator and target subspace

The projector onto $\mathcal{V}_i^\perp$ satisfies

$$
\Pi_i = I_n-V_iV_i^T, \qquad
\Pi_i^T = \Pi_i, \qquad
\Pi_i^2 = \Pi_i, \qquad
\mathrm{range}(\Pi_i)=\mathcal{V}_i^\perp.
$$

Define the symmetric projected operator and starting residual by

$$
A_{i,\perp} = \Pi_i\widehat{A}_i\Pi_i,
\qquad
A_{i,\perp}^T=A_{i,\perp},
$$

$$
y_i^{(0)}=0, \qquad
\widehat{r}_i^{(0)}=\widehat{b}_i-\widehat{A}_iy_i^{(0)}=\widehat{b}_i,
\qquad
r_{i,\perp}=\Pi_i\widehat{r}_i^{(0)}.
$$

For $m\geq1$, the target Krylov subspace is

$$
\mathcal{W}_{i,m}=\mathcal{K}_m(A_{i,\perp},r_{i,\perp})
=\mathrm{span}\lbrace
r_{i,\perp},\,
A_{i,\perp}r_{i,\perp},\,\ldots,\,
A_{i,\perp}^{m-1}r_{i,\perp}
\rbrace
\subseteq\mathcal{V}_i^\perp.
$$

The implementation returns a basis matrix $W=W_\ell$ with actual dimension $\ell$:

$$
W_\ell=[w_1,\ldots,w_\ell]\in\mathbb{R}^{n\times\ell},
\qquad
0\leq\ell\leq m_*:=\min(m,n-k).
$$

For a nonzero start and $\ell$ accepted Arnoldi vectors,

$$
\mathrm{range}(W_\ell)=\mathcal{K}_\ell(A_{i,\perp},r_{i,\perp}),
\qquad
V_i^TW_\ell=0, \qquad W_\ell^TW_\ell=I_\ell.
$$

### 3.2 Initialization and two-pass Arnoldi recurrence

The initial projection in [varvisc_build_projected_arnoldi.m](varvisc_build_projected_arnoldi.m) is evaluated twice:

$$
\rho^{(0)}=\widehat{b}_i, \qquad
\rho^{(s)}=\rho^{(s-1)}-V_i(V_i^T\rho^{(s-1)}),
\qquad s=1,2.
$$

Thus $\rho^{(2)}=r_{i,\perp}$ in exact arithmetic. If the start is accepted, set

$$
\beta=\lVert\rho^{(2)}\rVert_2, \qquad
w_1=\frac{\rho^{(2)}}{\beta}.
$$

At iteration $j$, let $W_j=[w_1,\ldots,w_j]$ and compute

$$
a_j=\widehat{A}_iw_j, \qquad z_j^{(0)}=a_j.
$$

For each orthogonalization pass $s=1,2$, evaluate in order

$$
c_j^{(s)}=V_i^Tz_j^{(s-1)}, \qquad
\widetilde{z}_j^{(s)}=z_j^{(s-1)}-V_ic_j^{(s)},
$$

$$
g_j^{(s)}=W_j^T\widetilde{z}_j^{(s)}, \qquad
z_j^{(s)}=\widetilde{z}_j^{(s)}-W_jg_j^{(s)}.
$$

Accumulate both passes into the Arnoldi coefficients:

$$
\gamma_j=c_j^{(1)}+c_j^{(2)}\in\mathbb{R}^k, \qquad
h_{1:j,j}=g_j^{(1)}+g_j^{(2)}\in\mathbb{R}^j,
\qquad
\eta_j=\lVert z_j^{(2)}\rVert_2.
$$

If $j<m_*$ and the remainder is accepted, append

$$
h_{j+1,j}=\eta_j, \qquad
w_{j+1}=\frac{z_j^{(2)}}{\eta_j}.
$$

These updates give the column relation

$$
\widehat{A}_iw_j=V_i\gamma_j+W_jh_{1:j,j}+z_j^{(2)}.
$$

Since $\Pi_iw_j=w_j$, the exact-arithmetic remainder also satisfies

$$
z_j^{(2)}=(I_n-W_jW_j^T)\Pi_i\widehat{A}_iw_j
         =(I_n-W_jW_j^T)A_{i,\perp}w_j.
$$

This identity establishes that the recurrence is Arnoldi applied to $A_{i,\perp}$ with starting vector $r_{i,\perp}$, implemented through products with $\widehat{A}_i$ and orthogonalization against $V_i$.

### 3.3 Termination and returned dimension

Let $\varepsilon_b=100\varepsilon_{\mathrm{mach}}$ be the default breakdown tolerance and let $\delta_{\min}$ denote MATLAB's `realmin`. The start is rejected, yielding $W\in\mathbb{R}^{n\times0}$, when

$$
m_*=0
\quad\text{or}\quad
\beta\leq\varepsilon_b\max(\lVert\widehat{b}_i\rVert_2,\delta_{\min}).
$$

After processing column $j$, the algorithm terminates with $\ell=j$ when

$$
j=m_*
\quad\text{or}\quad
\eta_j\leq\varepsilon_b\max(\lVert a_j\rVert_2,\delta_{\min}).
$$

If $\eta_j=0$ exactly, the generated Krylov subspace is invariant under $A_{i,\perp}$. A small positive $\eta_j$ triggers numerical breakdown without asserting exact invariance. The default requested dimension is $m=20$; rejected or unavailable directions are never replaced by random padding.

### 3.4 Matrix Arnoldi identities

For $\ell>0$, define

$$
B_\ell=[\gamma_1,\ldots,\gamma_\ell]\in\mathbb{R}^{k\times\ell},
\qquad
H_\ell=(h_{pq})_{p,q=1}^{\ell}\in\mathbb{R}^{\ell\times\ell},
\qquad
f_\ell=z_\ell^{(2)}.
$$

Here $H_\ell$ is upper Hessenberg, with the accumulated coefficients and subdiagonal entries defined above. Let $e_\ell$ be the last coordinate vector in $\mathbb{R}^\ell$. The retained full-operator and projected-operator relations are

$$
\widehat{A}_iW_\ell
=V_iB_\ell+W_\ell H_\ell+f_\ell e_\ell^T,
$$

$$
A_{i,\perp}W_\ell
=W_\ell H_\ell+f_\ell e_\ell^T,
\qquad
V_i^Tf_\ell=0, \qquad W_\ell^Tf_\ell=0.
$$

In exact arithmetic,

$$
B_\ell=V_i^T\widehat{A}_iW_\ell, \qquad
H_\ell=W_\ell^T\widehat{A}_iW_\ell
      =W_\ell^TA_{i,\perp}W_\ell.
$$

Because $A_{i,\perp}$ is symmetric, $H_\ell$ is symmetric and therefore tridiagonal in exact arithmetic. The code retains the full small matrix to accommodate rounding error and reorthogonalization.

The augmented basis passed to the coarse correction is

$$
S_i=[V_i\;W_\ell], \qquad
S_i^TS_i=I_{k+\ell}, \qquad
\mathrm{range}(S_i)=\mathcal{V}_i\oplus\mathcal{K}_\ell(A_{i,\perp},r_{i,\perp}).
$$

For $\ell=0$, set $S_i=V_i$. The Krylov term is rebuilt from the current operator and RHS at every system index $i$; it is not added to the cached physical basis $U$.

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
