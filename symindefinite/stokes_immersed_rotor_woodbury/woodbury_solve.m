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
%   preconditioner.  What it is exposed to in floating point is the conditioning
%   of Cap, which is why cond(Cap) is reported alongside every solve: the forward
%   error tracks it, and ||dC||_F/||C||_F reaches O(1) over a rotor revolution.
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
%   WHY BOTH OFF-DIAGONAL BLOCKS ARE COMPUTED.  dC'*YSel and Sel'*Y_dC are
%   transposes of one another in exact arithmetic, so one could be obtained from
%   the other for free.  They are computed INDEPENDENTLY (two small GEMMs) so that
%   info.cap_symres is a real check on the wiring rather than a tautology --
%   an asymmetry there means Y_dC, YSel or Sel is wrong, not that rounding
%   occurred.
%
%   THE dC == 0 BRANCH IS ALGEBRA, NOT AN OPTIMIZATION.  When the coupling does
%   not move (disk_static, and trivially at n == ref), dC is exactly zero, so
%   U B U' = 0 * Sel' + Sel * 0' = 0 and therefore K_n == K_1 EXACTLY.  The
%   correction term is provably zero, so it is skipped rather than computed and
%   rounded -- which is what lets the falsification control assert bit-for-bit
%   agreement with the uncorrected frozen inverse.  Cap and its diagnostics are
%   still formed, so the reported cost and conditioning stay representative of
%   the general path.
%
%   A NOTE ON BIT-EXACTNESS.  Because the RHS rides along in the batched
%   multi-column apply, K_1^{-1}b as computed here can differ in its last bits
%   from a standalone single-column solve of b -- different BLAS blocking, same
%   answer to ~1e-15.  So "the correction was skipped" must be asserted via
%   info.correction_norm == 0, not by comparing against a separate frozen solve.
%
%   INFO: .cap_cond .cap_smin .cap_smax .cap_rcond .cap_symres .Cap
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
    y    = Z(:, nC + 1);

    % --- Cap = B + U'Y0, assembled blockwise ------------------------------
    %   [ dC'Y_dC        dC'YSel + I ]
    %   [ Sel'Y_dC + I   Sel'YSel    ]   <- lower right is precomputed & constant
    Ic  = eye(nC);
    G11 = full(dC' * Y_dC);
    G12 = full(dC' * ctx.YSel);
    G21 = full(ctx.Sel' * Y_dC);
    Cap_raw = [G11, G12 + Ic; G21 + Ic, ctx.SelYSel];

    cap_symres = norm(Cap_raw - Cap_raw', 'fro') / ...
                 max(norm(Cap_raw, 'fro'), eps);
    Cap = (Cap_raw + Cap_raw') / 2;

    % --- Conditioning: free at 2nC <= 88, and the metric the error tracks --
    % Timed separately: svd+rcond are DIAGNOSTICS a production solve would not
    % run, so folding them into the headline cost would bias the comparison
    % against the method.  info.t_diag lets the caller subtract them.
    t_diag_0 = tic;
    sv       = svd(Cap);
    cap_smax = sv(1);
    cap_smin = sv(end);
    if cap_smin > 0
        cap_cond = cap_smax / cap_smin;
    else
        cap_cond = Inf;
    end
    cap_rcond = rcond(Cap);
    t_diag    = toc(t_diag_0);

    if ~isfinite(cap_rcond) || cap_rcond < eps
        warning('woodbury_solve:singularCapacitance', ...
                ['step %d: rcond(Cap) = %.3e is at or below eps.  K_n is ' ...
                 'nonsingular iff Cap is, so this is a genuine breakdown of ' ...
                 'the update, not a tolerance to be raised.'], n, cap_rcond);
    end

    % --- The solve ---------------------------------------------------------
    dC_is_zero = (nnz(dC) == 0);
    if dC_is_zero
        x = y;                                       % K_n == K_1 exactly
    else
        w = Cap \ [full(dC' * y); full(ctx.Sel' * y)];
        x = y - Y_dC * w(1:nC) - ctx.YSel * w(nC+1:end);
    end

    info                  = struct();
    info.cap_cond         = cap_cond;
    info.cap_smin         = cap_smin;
    info.cap_smax         = cap_smax;
    info.cap_rcond        = cap_rcond;
    info.cap_symres       = cap_symres;
    info.Cap              = Cap;
    info.dC_normF         = norm(dC, 'fro');
    info.dC_rel           = info.dC_normF / max(ctx.Cblk_ref_normF, eps);
    info.dC_is_zero       = dC_is_zero;
    % How far the Woodbury term actually moved the iterate.  Exactly 0 when the
    % dC == 0 branch is taken -- that is the sharp form of the falsification
    % control, and it is a property of THIS solve, not of two BLAS paths agreeing.
    info.correction_norm  = norm(x - y);
    info.correction_rel   = info.correction_norm / max(norm(x), eps);
    info.n_backsolves     = size(dC, 2);             % operator update: nC
    info.n_backsolves_rhs = 1;                       % the right-hand side
    info.t_solve          = toc(t0);                 % everything, incl. diagnostics
    info.t_diag           = t_diag;                  % the svd+rcond part alone
    info.t_net            = info.t_solve - t_diag;   % the algorithmic cost
end
