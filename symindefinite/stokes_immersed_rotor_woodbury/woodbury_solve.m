function [x, info] = woodbury_solve(ctx, S, n, b)
%WOODBURY_SOLVE  Solve K_n x = b using only the frozen factorization of A_1 = K_1.
%   [X, INFO] = WOODBURY_SOLVE(CTX, S, N, B)
%
%   CTX from woodbury_context_init, S from build_stokes_sequence.  With
%   U = [dC, Sel], dC = Cblk_n - Cblk_ref, B = [0 I; I 0] and Y0 = K_1^{-1}U:
%
%       Cap = B + U' Y0                                   (2nC-by-2nC, symmetric)
%       x   = K_1^{-1}b - Y0 * ( Cap \ (U' K_1^{-1} b) )
%
%   This is EXACT in exact arithmetic -- it is not an approximation or a
%   preconditioner.  Any error it makes is therefore an error of EVALUATION.
%
%   THIS IS A NAIVE EVALUATION, ON PURPOSE.  The expression is written out in the
%   order given above with no defensive measures: Cap is NOT symmetrized, C^{-1} is
%   formed explicitly, U and Y0 are not orthogonalized, there is no iterative
%   refinement, and the final subtraction x = z - Y0*w is unguarded -- including
%   when dC is exactly zero and the correction is provably zero, which is computed
%   and rounded like any other step.  A "stabilized" variant would make the reported
%   accuracy a property of the defenses rather than of the problem, and the point of
%   this study is the problem.  See tests/test_stress_metrics.m for the two
%   constructed systems on which this same evaluation loses every digit.
%
%   WHAT IT IS EXPOSED TO is cancellation, in two places, both reported per step:
%     info.cancel_cap  (||C^{-1}|| + ||U|| ||Y0||) / ||Cap||   -- the small matrix
%     info.cancel_sub  (||z|| + ||Y0 w||) / ||z - Y0 w||       -- the subtraction
%   Neither is visible in cond(K_n).  cond(Cap) is reported too, but the measured
%   error follows the cancellation factors, not the condition numbers.
%
%   COST.  nC backsolves for Y_dC = K_1^{-1}dC, plus ONE for the right-hand side.
%   The Sel half of Y0 is time-independent and was solved once in
%   woodbury_context_init, so this is nC and not 2nC -- test_woodbury_identity T5
%   pins that.  The 2nC-by-2nC dense solve is negligible (nC <= 44 here).
%
%   All nC+1 columns go through woodbury_apply_ref in a SINGLE call: the sparse
%   triangular solves batch across columns, and the per-call overhead (two
%   diagonal scalings, one permutation gather/scatter) is then paid once instead
%   of twice.
%
%   WHY Cap IS NOT SYMMETRIZED.  Cap = C^{-1} + U'Y0 is symmetric in exact
%   arithmetic, so (Cap + Cap')/2 would be free accuracy -- and is exactly the kind
%   of quiet repair this study must not make.  The asymmetry is MEASURED instead
%   (info.cap_symres) and left in place.  It doubles as a wiring check: the two
%   off-diagonal blocks dC'*YSel and Sel'*Y_dC are formed by independent sums, so a
%   symres above rounding level means Y_dC, YSel or Sel is wrong.
%
%   A NOTE ON BIT-EXACTNESS.  Because the RHS rides along in the batched
%   multi-column apply, K_1^{-1}b as computed here can differ in its last bits
%   from a standalone single-column solve of b -- different BLAS blocking, same
%   answer to ~1e-15.  Likewise, since the dC == 0 case is no longer special-cased,
%   info.correction_norm there is ~1e-16 rather than exactly 0.
%
%   NORMS.  The tall factors U and Y0 are ntot-by-2nC, so cancel_cap uses Frobenius
%   norms (O(n*nC)) rather than spectral ones (an svd per step).  The two differ by
%   at most sqrt(2nC) ~ 9, immaterial for a quantity read against 1 or 1e14.
%
%   INFO: .cap_cond .cap_smin .cap_smax .cap_rcond .cap_symres .Cap
%         .cancel_cap .cancel_sub .rho          the cancellation diagnostics
%         .dC_normF .dC_rel .dC_is_zero .correction_norm .correction_rel
%         .n_backsolves .n_backsolves_rhs
%         .t_solve (all of it) .t_diag (the svd/rcond part) .t_net (the rest)
%
%   See also: woodbury_context_init, seq_dCblk, seq_K.

    t0 = tic;
    nC = ctx.nC;

    [~, dC] = seq_dCblk(S, n, ctx.ref);

    % --- The only per-step backsolves: nC update columns + the RHS, batched -
    Z    = woodbury_apply_ref(ctx, [full(dC), b]);
    Y_dC = Z(:, 1:nC);
    z    = Z(:, nC + 1);                             % z = A^{-1} b

    % --- The identity, written out -----------------------------------------
    Zc   = zeros(nC);
    Ic   = eye(nC);
    U    = [full(dC), full(ctx.Sel)];                % U, and V = U'
    Y0   = [Y_dC, ctx.YSel];                         % Y0 = A^{-1} U
    Cinv = [Zc, Ic; Ic, Zc];                         % C^{-1} = C = B, exactly

    Cap = Cinv + U' * Y0;                            % Cap = C^{-1} + V A^{-1} U
    w   = Cap \ (U' * z);
    Yw  = Y0 * w;
    x   = z - Yw;                                    % the subtraction, unguarded

    % --- Diagnostics: measured, never fed back ------------------------------
    % Timed separately: a production solve would run none of these, so folding
    % them into the headline cost would bias the comparison against the method.
    % info.t_diag lets the caller subtract them.
    t_diag_0 = tic;

    cap_symres = norm(Cap - Cap', 'fro') / max(norm(Cap, 'fro'), eps);

    sv       = svd(Cap);
    cap_smax = sv(1);
    cap_smin = sv(end);
    if cap_smin > 0
        cap_cond = cap_smax / cap_smin;
    else
        cap_cond = Inf;
    end
    cap_rcond = rcond(Cap);

    % The two places digits can go missing.  Both are 1 when nothing cancels.
    cancel_cap = (norm(Cinv, 'fro') + norm(U, 'fro') * norm(Y0, 'fro')) / ...
                 max(norm(Cap, 'fro'), realmin);
    cancel_sub = (norm(z) + norm(Yw)) / max(norm(x), realmin);
    rho        = norm(z) / max(norm(x), realmin);

    t_diag = toc(t_diag_0);

    if ~isfinite(cap_rcond) || cap_rcond < eps
        warning('woodbury_solve:singularCapacitance', ...
                ['step %d: rcond(Cap) = %.3e is at or below eps.  K_n is ' ...
                 'nonsingular iff Cap is, so this is a genuine breakdown of ' ...
                 'the update, not a tolerance to be raised.'], n, cap_rcond);
    end

    dC_is_zero = (nnz(dC) == 0);

    info                  = struct();
    info.cap_cond         = cap_cond;
    info.cap_smin         = cap_smin;
    info.cap_smax         = cap_smax;
    info.cap_rcond        = cap_rcond;
    info.cap_symres       = cap_symres;
    info.Cap              = Cap;
    info.cancel_cap       = cancel_cap;
    info.cancel_sub       = cancel_sub;
    info.rho              = rho;
    info.dC_normF         = norm(dC, 'fro');
    info.dC_rel           = info.dC_normF / max(ctx.Cblk_ref_normF, eps);
    info.dC_is_zero       = dC_is_zero;
    % How far the Woodbury term actually moved the iterate.  When dC == 0 the
    % correction is provably zero but is still computed, so this lands at ~1e-16
    % rather than exactly 0 -- see the naive-evaluation note above.
    info.correction_norm  = norm(Yw);
    info.correction_rel   = info.correction_norm / max(norm(x), eps);
    info.n_backsolves     = size(dC, 2);             % operator update: nC
    info.n_backsolves_rhs = 1;                       % the right-hand side
    info.t_solve          = toc(t0);                 % everything, incl. diagnostics
    info.t_diag           = t_diag;                  % the svd+rcond part alone
    info.t_net            = info.t_solve - t_diag;   % the algorithmic cost
end
