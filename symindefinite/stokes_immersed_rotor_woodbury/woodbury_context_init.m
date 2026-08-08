function ctx = woodbury_context_init(S)
%WOODBURY_CONTEXT_INIT  Factorize A_1 = K_1 once and cache the time-independent half.
%   CTX = WOODBURY_CONTEXT_INIT(S)   with S from build_stokes_sequence.
%
%   This is the ONE factorization the whole study is allowed.  Everything the
%   Woodbury update needs that does not depend on the timestep is computed here,
%   so that a per-step solve costs nC backsolves rather than 2nC.
%
%   THE ALGEBRA.  build_stokes_sequence guarantees (and asserts every step)
%
%       K_n = K_1 + U B U',    U = [dC, Sel],   dC = Cblk_n - Cblk_1,
%                              B = [0 I; I 0]   (2nC-by-2nC, B^{-1} = B),
%
%   so Woodbury gives, with Y0 = K_1^{-1} U = [Y_dC, YSel]:
%
%       K_n^{-1} b = K_1^{-1}b  -  Y0 * ( Cap \ (U' K_1^{-1} b) ),
%       Cap = B^{-1} + U' K_1^{-1} U = B + U' Y0.
%
%   WHAT IS TIME-INDEPENDENT.  Sel = [0; 0; I_nC] selects the multiplier rows and
%   never moves, so the second half of Y0,
%
%       YSel = K_1^{-1} Sel,
%
%   and with it the whole lower-right block Sel' K_1^{-1} Sel of Cap, are constant
%   across the sequence.  They are solved here, once, for nC backsolves.  Only the
%   dC half is rebuilt per step.  (lowrank_update_basis uses the same ctx.YSel
%   trick for its span computation.)
%
%   THE FACTORS ARE STORED RAW, NOT AS A decomposition OBJECT, and applied by
%   woodbury_apply_ref -- which is 27x faster on this operator.  See that file for
%   the measurement and why the choice changes the study's conclusion.
%
%   THE REFERENCE IS STEP 1, HARD-CODED.  There is no refresh cadence, on purpose:
%   the question is whether ONE factorization of A_1 can serve the whole sequence,
%   and a refresh knob would let that question be answered by refactorizing.
%
%   Output CTX:
%     .ref .ntot .nC              the reference step and dimensions
%     .L .dD .perm .Sscale        the frozen ldl factors (apply: woodbury_apply_ref)
%     .nnzK1 .nnzL .fill_ratio    size of the factorization on record
%     .YSel .SelYSel              the cached time-independent blocks
%     .Sel                        S.Sel, kept so callers need not carry S around
%     .Cblk_ref_normF             denominator for the dC_rel drift measure
%     .n_backsolves_setup         nC, the one-time cost
%     .t_factor .t_setup          seconds: the factorization alone, and the total
%
%   See also: woodbury_apply_ref, woodbury_solve, seq_K, seq_dCblk.

    t0  = tic;
    ref = 1;

    K1 = seq_K(S, ref);

    t_fac = tic;
    [L, D, perm, Sscale] = ldl(K1, 'vector');
    % D is block diagonal (1x1 and 2x2 pivots), so this factorization is cheap and
    % is what makes the 2x2 pivots usable without unpacking them by hand.
    dD       = decomposition(D);
    t_factor = toc(t_fac);

    ctx = struct();
    ctx.ref     = ref;
    ctx.ntot    = S.n;
    ctx.nC      = S.nC;
    ctx.L       = L;
    ctx.dD      = dD;
    ctx.perm    = perm;
    ctx.Sscale  = Sscale;
    ctx.nnzK1   = nnz(K1);
    ctx.nnzL    = nnz(L);
    ctx.fill_ratio = nnz(L) / max(nnz(K1), 1);

    % nC backsolves, paid once: Sel does not move, so neither does K_1^{-1}Sel.
    YSel    = woodbury_apply_ref(ctx, full(S.Sel));
    SelYSel = full(S.Sel' * YSel);
    SelYSel = (SelYSel + SelYSel') / 2;     % symmetric by construction

    ctx.YSel               = YSel;
    ctx.SelYSel            = SelYSel;
    ctx.Sel                = S.Sel;
    ctx.Cblk_ref_normF     = norm(S.Cblk{ref}, 'fro');
    ctx.n_backsolves_setup = size(S.Sel, 2);
    ctx.t_factor           = t_factor;
    ctx.t_setup            = toc(t0);

    % The applier is the load-bearing piece of the cost argument, so its
    % correctness is checked here rather than only in the test suite: a wrong
    % permutation or scaling would still produce plausible-looking numbers.
    relres = norm(K1 * YSel - full(S.Sel), 'fro') / ...
             max(norm(full(S.Sel), 'fro'), eps);
    if ~(relres < 1e-8)
        error('woodbury_context_init:badApply', ...
              ['woodbury_apply_ref does not invert K_1 (||K_1 Y - Sel||/||Sel|| ' ...
               '= %.3e).  The ldl permutation/scaling convention has changed.'], ...
              relres);
    end
    ctx.apply_relres = relres;
end
