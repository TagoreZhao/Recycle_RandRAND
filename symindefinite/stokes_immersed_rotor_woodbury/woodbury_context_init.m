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
%   is constant across the sequence.  It is solved here, once, for nC backsolves;
%   only the dC half is rebuilt per step.  That is a saving in BACKSOLVES, which is
%   what the cost claim is about, and it changes no arithmetic: the same frozen
%   factors are applied to the same columns either way.  (lowrank_update_basis uses
%   the same ctx.YSel trick for its span computation.)
%
%   ctx.SelYSel = Sel' YSel is cached as well, but woodbury_solve no longer reads
%   it -- it forms all of U'Y0 in one GEMM, naively.  It is kept for the tests that
%   check the cached block against a fresh product.
%
%   THE FACTORS ARE STORED RAW, NOT AS A decomposition OBJECT, and applied by
%   woodbury_apply_ref -- which is 27x faster on this operator.  See that file for
%   the measurement and why the choice changes the study's conclusion.
%
%   build_stokes_sequence now guarantees a finite sequence (it refuses to return
%   or cache a step whose K \ b is non-finite), so a singular K_1 reaching this
%   function means the REFERENCE STEP itself is degenerate -- and the error below
%   says so rather than blaming the applier.
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
    % NOT symmetrized, though it is symmetric in exact arithmetic: woodbury_solve
    % is a naive evaluation and must not be handed pre-repaired inputs.  Its
    % asymmetry is what info.cap_symres measures.

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
        % Two different faults produce this same symptom, and they want opposite
        % responses.  Name the one that is actually present: blaming the applier
        % for a singular K_1 once sent a debugging session 60 steps upstream of
        % the real defect.  Nothing here runs on the success path.
        n_bad = nnz(~isfinite(YSel));
        ws    = warning('off', 'MATLAB:singularMatrix');
        rcD   = 1 / condest(D);
        warning(ws);
        if n_bad > 0 || ~isfinite(rcD) || rcD < eps
            error('woodbury_context_init:singularReference', ...
                  ['K_1 is numerically singular (1/condest(D) = %.3e, %d ' ...
                   'non-finite entries in K_1^{-1}Sel), so there is nothing here ' ...
                   'for the applier to be right about.  K_1 = S P L D L'' P'' S ' ...
                   'with L unit lower triangular and P, S invertible, so K_1 is ' ...
                   'singular exactly when D is -- the fault is in the OPERATOR, ' ...
                   'not in the ldl convention and not in this file.  The usual ' ...
                   'cause is a rank-deficient coupling block C_1; ' ...
                   'build_stokes_sequence''s assert_coupling_feasible check names ' ...
                   'it at assembly time when the row count alone reveals it.'], ...
                  rcD, n_bad);
        end
        error('woodbury_context_init:badApply', ...
              ['woodbury_apply_ref does not invert K_1 (||K_1 Y - Sel||/||Sel|| ' ...
               '= %.3e).  The ldl permutation/scaling convention has changed.'], ...
              relres);
    end
    ctx.apply_relres = relres;
end
