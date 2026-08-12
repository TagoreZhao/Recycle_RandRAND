function X = frozen_ldl_apply(ctx, B)
%FROZEN_LDL_APPLY  Apply the frozen inverse: X = K_ref^{-1} B, one or many columns.
%
%   X = FROZEN_LDL_APPLY(CTX, B)   CTX from frozen_ldl_context.
%
%   With [L, D, p, S] = ldl(K_ref, 'vector') MATLAB guarantees
%
%       S*K_ref*S = P L D L' P'   (P the permutation with P'x = x(p)),
%
%   so    K_ref^{-1} B = S * scatter_p( L^{-T} D^{-1} L^{-1} ( (S*B)(p, :) ) ).
%
%   PASS ALL COLUMNS IN ONE CALL.  The sparse triangular solves below batch (the
%   fixed per-call cost -- two diagonal scalings and the permutation
%   gather/scatter -- is paid once per call, not once per column), which is the
%   whole reason (2q+1)*k backsolves against a frozen factor is a cheap way to
%   build a coarse space.  A decomposition object would NOT batch here and would
%   turn the same algebra into k separate solves; the measurement behind that
%   claim, and what it did to a published cost conclusion, is documented in
%   stokes_immersed_rotor_woodbury/woodbury_apply_ref.m, of which this is the
%   plain-matrix twin (that one is bound to a build_stokes_sequence context, this
%   one only needs a matrix).
%
%   See also: frozen_ldl_context, build_lowrank_sketch_V.

    if issparse(B), B = full(B); end

    Y = ctx.Sscale * B;
    Y = Y(ctx.perm, :);
    Y = ctx.L \ Y;
    Y = ctx.dD \ Y;
    Y = ctx.L' \ Y;

    X = zeros(size(B));
    X(ctx.perm, :) = Y;
    X = ctx.Sscale * X;
end
