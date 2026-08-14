function ctx = varvisc_esketch_ref_context(K)
%ESKETCH_REF_CONTEXT  Factorize the reference KKT matrix ONCE into its exact
% split factor and freeze it.
%
%   ctx = VARVISC_ESKETCH_REF_CONTEXT(K)
%
%   The one factorization the E-sketch arm is allowed.  Everything later steps
%   need from A_1 = K is stored here: the exact split factor C_ref (as a
%   make_ildl_precond struct, whose applyCinv / applyCtinv handles are BATCHED
%   -- multi-column right-hand sides go through the sparse triangular solves in
%   one call) and the symmetrized matrix ITSELF, which is what makes the
%   difference dK = A_2 - A_1 formable at a later step without re-assembling A_1.
%
%   WHY make_ildl_precond('exact') AND NOT RAW ldl FACTORS.  The sketch target is
%   E = C_ref^{-1} dK C_ref^{-T} with C_ref C_ref' = |A_1|, i.e. the SAME split
%   factor family the preconditioner under MINRES uses -- so the |D|^{1/2}
%   handling of 1x1/2x2 pivots must be make_ildl_precond's, not a re-derivation.
%   Its handles apply C_ref^{-1} and C_ref^{-T} directly, which is all the
%   builder needs; K_ref^{-1} itself is never applied.
%
%   K IS SYMMETRIZED and the symmetrized matrix is what is stored, matching what
%   make_ildl_precond factors internally, so dK = K_n - ctx.Kref is exactly
%   A_2 - A_1 for the A_1 that was factored.  The asymmetry being removed is
%   assembly round-off (apply_dirichlet_sym eliminates symmetrically), not a
%   modelling choice.
%
%   ctx fields:
%     .Kref                     the symmetrized reference matrix, as factored
%     .P                        exact make_ildl_precond struct (C_ref handles)
%     .n                        size(Kref,1) -- the shape guard `cached` checks
%     .nnzK .nnzL .fill_ratio   size of the factorization, for cost accounting
%     .t_factor                 seconds spent in ldl
%
%   See also: varvisc_build_Esketch_V, src.precond.make_ildl_precond.

    Ksym = (K + K') / 2;

    t0 = tic;
    P  = src.precond.make_ildl_precond(Ksym, struct('mode', 'exact'));
    t_factor = toc(t0);

    ctx            = struct();
    ctx.Kref       = Ksym;
    ctx.P          = P;
    ctx.n          = size(Ksym, 1);
    ctx.nnzK       = nnz(Ksym);
    ctx.nnzL       = P.nnzL;
    ctx.fill_ratio = P.fill_ratio;
    ctx.t_factor   = t_factor;
end
