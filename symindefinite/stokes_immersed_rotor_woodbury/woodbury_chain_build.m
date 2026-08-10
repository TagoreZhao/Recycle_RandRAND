function lev = woodbury_chain_build(ctx, S, nlevel)
%WOODBURY_CHAIN_BUILD  Build a chain of Woodbury updates, each on top of the last.
%   LEV = WOODBURY_CHAIN_BUILD(CTX, S, NLEVEL)
%
%   This is the "recursively reuse the computed inverse" scheme, the thing the
%   production path deliberately does NOT do.  Level 1 is CTX, the frozen
%   factorization of K_1.  Level k treats the level k-1 OPERATOR as its A^{-1} and
%   corrects it by the INCREMENTAL update
%
%       K_k = K_{k-1} + U_k B U_k',   U_k = [Cblk_k - Cblk_{k-1}, Sel]
%
%   so that applying level k costs k-1 nested corrections.  Because the increments
%   telescope, the level-n operator and the production scheme's single update from
%   K_1 represent the SAME matrix inverse -- any difference between them is
%   floating point and nothing else.  That is what makes the comparison meaningful.
%
%   COST.  Building level k applies level k-1 to 2nC columns, so the chain costs
%   O(NLEVEL^2 * nC) base backsolves and O(NLEVEL) memory per level.  It is a
%   diagnostic, not a method -- see run_woodbury_recursive for why the production
%   path re-anchors instead.
%
%   Each entry of LEV has .U (ntot-by-2nC), .Y = X_{k-1} U_k, and .Cap.  LEV(1) is
%   empty: level 1 is CTX itself.  Apply with woodbury_chain_apply.
%
%   The arithmetic per level is the same naive expression as woodbury_solve: no
%   symmetrization, C^{-1} formed explicitly, nothing re-orthogonalized.
%
%   See also: woodbury_chain_apply, run_woodbury_recursive, woodbury_solve.

    nC   = ctx.nC;
    Cinv = [zeros(nC), eye(nC); eye(nC), zeros(nC)];   % C^{-1} = C = B, exactly

    lev = struct('U', cell(nlevel, 1), 'Y', cell(nlevel, 1), 'Cap', cell(nlevel, 1));
    for k = 2:nlevel
        [~, dCk] = seq_dCblk(S, k, k - 1);             % INCREMENTAL, not vs step 1
        Uk = [full(dCk), full(S.Sel)];
        Yk = woodbury_chain_apply(ctx, lev, k - 1, Uk);
        lev(k).U   = Uk;
        lev(k).Y   = Yk;
        lev(k).Cap = Cinv + Uk' * Yk;
    end
end
