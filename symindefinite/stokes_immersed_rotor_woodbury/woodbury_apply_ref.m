function X = woodbury_apply_ref(ctx, B)
%WOODBURY_APPLY_REF  Apply the frozen inverse: X = K_1^{-1} B, one or many columns.
%   X = WOODBURY_APPLY_REF(CTX, B)   CTX from woodbury_context_init.
%
%   With [L, D, p, S] = ldl(K_1, 'vector') MATLAB guarantees
%
%       S*K_1*S = P L D L' P'   (P the permutation with P'x = x(p)),
%
%   so    K_1^{-1} B = S * scatter_p( L^{-T} D^{-1} L^{-1} ( (S*B)(p, :) ) ).
%
%   WHY THE FACTORS ARE APPLIED BY HAND RATHER THAN VIA decomposition().  Measured
%   on this operator (h0 = 0.05, n = 5840, nnz(L) = 327k, nC = 20 columns):
%
%       decomposition(K_1) \ B      47.6 ms
%       these five lines              1.8 ms      <-- 27x faster, same answer to 6e-15
%
%   decomposition's mldivide charges a large per-column overhead and does not
%   batch, so a multi-column right-hand side costs it nC times a single solve.
%   The sparse triangular solves below DO batch (20 columns cost 4.4x one column,
%   not 20x), and batching is the whole reason the Woodbury update is cheap here.
%   Using decomposition would have made the method look ~5x SLOWER than
%   refactorizing when it is in fact several times faster -- an artifact of the
%   API, not a property of the algebra.
%
%   B may have any number of columns; pass them all in ONE call, since the fixed
%   per-call cost (the two diagonal scalings and the permutation gather/scatter)
%   is paid once per call rather than once per column.
%
%   See also: woodbury_context_init, woodbury_solve.

    Y = ctx.Sscale * B;
    Y = Y(ctx.perm, :);
    Y = ctx.L \ Y;
    Y = ctx.dD \ Y;
    Y = ctx.L' \ Y;

    X = zeros(size(B));
    X(ctx.perm, :) = Y;
    X = ctx.Sscale * X;
end
