function ctx = varvisc_frozen_ldl_context(K)
%FROZEN_LDL_CONTEXT  Factorize the reference KKT matrix ONCE and freeze it.
%
%   ctx = FROZEN_LDL_CONTEXT(K)
%
%   The one factorization the low-rank sketch arm is allowed.  Everything later
%   steps need from A_1 = K is stored here: the raw ldl factors (applied by
%   varvisc_frozen_ldl_apply) and the matrix ITSELF, which is what makes the difference
%   operator dK = A_2 - A_1 formable at a later step without re-assembling A_1.
%
%   THE FACTORS ARE STORED RAW, NOT AS A decomposition OBJECT.  decomposition's
%   mldivide does not batch a multi-column right-hand side -- k columns cost k
%   times a single solve -- and this method's whole cost argument is (2q+1)*k
%   BATCHED backsolves against a frozen factor.  Measured on this operator, the
%   difference is 27x; see the measurement and the reasoning in
%   stokes_immersed_rotor_woodbury/woodbury_apply_ref.m, which varvisc_frozen_ldl_apply
%   mirrors.  (D, the block-diagonal pivot matrix, IS wrapped in a decomposition:
%   it is block diagonal with 1x1 and 2x2 pivots, so the wrapper is what makes the
%   2x2 pivots usable without unpacking them by hand, and it costs nothing.)
%
%   K IS SYMMETRIZED BEFORE FACTORIZATION and the symmetrized matrix is what is
%   stored, so ctx.Kref is exactly the matrix ctx's factors invert.  Two things
%   depend on that: dK = K_n - ctx.Kref is then exactly A_2 - A_1 for the A_1 that
%   was factored, and the sketch's transpose apply uses K_ref^{-T} = K_ref^{-1}.
%   The asymmetry being removed is assembly round-off (the engine's
%   apply_dirichlet_sym eliminates symmetrically), not a modelling choice.
%
%   ctx fields:
%     .Kref                     the symmetrized reference matrix, as factored
%     .L .dD .perm .Sscale      the frozen factors (apply: varvisc_frozen_ldl_apply)
%     .n                        size(Kref,1) -- the shape guard `cached` checks
%     .nnzK .nnzL .fill_ratio   size of the factorization, for cost accounting
%     .t_factor                 seconds spent in ldl
%
%   See also: varvisc_frozen_ldl_apply, varvisc_build_lowrank_sketch_V, woodbury_apply_ref.

    Ksym = (K + K') / 2;

    t0 = tic;
    [L, D, perm, Sscale] = ldl(Ksym, 'vector');
    t_factor = toc(t0);

    ctx            = struct();
    ctx.Kref       = Ksym;
    ctx.L          = L;
    ctx.dD         = decomposition(D);
    ctx.perm       = perm;
    ctx.Sscale     = Sscale;
    ctx.n          = size(Ksym, 1);
    ctx.nnzK       = nnz(Ksym);
    ctx.nnzL       = nnz(L);
    ctx.fill_ratio = nnz(L) / max(nnz(Ksym), 1);
    ctx.t_factor   = t_factor;
end
