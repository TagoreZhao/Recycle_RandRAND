function X = woodbury_chain_apply(ctx, lev, kmax, B)
%WOODBURY_CHAIN_APPLY  Apply the level-KMAX recursive inverse to the columns of B.
%   X = WOODBURY_CHAIN_APPLY(CTX, LEV, KMAX, B)
%
%   Level 1 is the frozen factorization of K_1.  Each level above it is one naive
%   Woodbury correction whose A^{-1} is the level below, so this is a loop and not
%   an actual recursion -- level k's correction is applied to level k-1's output.
%
%   KMAX <= 1 returns the frozen solve unchanged, which is what makes
%   woodbury_chain_build's own bootstrap work.
%
%   See also: woodbury_chain_build, woodbury_apply_ref, run_woodbury_recursive.

    X = woodbury_apply_ref(ctx, B);
    for k = 2:kmax
        X = X - lev(k).Y * (lev(k).Cap \ (lev(k).U' * X));
    end
end
