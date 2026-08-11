function ctx = woodbury_context_init(S, ref)
%WOODBURY_CONTEXT_INIT  Factorize A_ref = K_ref once and cache the time-independent half.
%   CTX = WOODBURY_CONTEXT_INIT(S)        with S from build_stokes_sequence.
%   CTX = WOODBURY_CONTEXT_INIT(S, REF)   freeze step REF instead of step 1.
%
%   This is the ONE factorization the whole study is allowed.  Everything the
%   Woodbury update needs that does not depend on the timestep is computed here,
%   so that a per-step solve costs nC backsolves rather than 2nC.
%
%   THE ALGEBRA.  build_stokes_sequence guarantees (and asserts every step)
%
%       K_n = K_r + U B U',    U = [dC, Sel],   dC = Cblk_n - Cblk_r,
%                              B = [0 I; I 0]   (2nC-by-2nC, B^{-1} = B),
%
%   for ANY pair of steps n, r -- so the algebra below holds at every REF, not
%   only at r = 1.  With Y0 = K_r^{-1} U = [Y_dC, YSel], Woodbury gives:
%
%       K_n^{-1} b = K_r^{-1}b  -  Y0 * ( Cap \ (U' K_r^{-1} b) ),
%       Cap = B^{-1} + U' K_r^{-1} U = B + U' Y0.
%
%   WHAT IS TIME-INDEPENDENT.  Sel = [0; 0; I_nC] selects the multiplier rows and
%   never moves, so the second half of Y0,
%
%       YSel = K_r^{-1} Sel,
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
%   or cache a step whose K \ b is non-finite), so a singular K_ref reaching this
%   function means the REFERENCE STEP itself is degenerate -- and the error below
%   says so rather than blaming the applier.
%
%   THE REFERENCE DEFAULTS TO STEP 1, and REF only moves WHICH SINGLE STEP is
%   frozen.  It is not a refresh cadence: there is still exactly one factorization
%   per context and no path anywhere that refactorizes mid-run, so the question
%   this study asks -- can ONE factorization serve the whole sequence -- is
%   untouched.  What REF buys is the experiment that question needs a control for:
%   hold the targets fixed and vary only the anchor.  test_reference_index R7 pins
%   the cost invariant against REF so the distinction stays a property of the code
%   rather than a claim in this comment.
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
%     .rcond_D                    1/condest(D): the conditioning the gate below is
%                                 written in, reported on the SUCCESS path too
%     .cond_ref                   condest(K_ref), NaN unless the gate had to look
%     .apply_relres               ||K_ref YSel - Sel||_F / ||Sel||_F
%
%   See also: woodbury_apply_ref, woodbury_solve, seq_K, seq_dCblk.

    t0 = tic;
    if nargin < 2 || isempty(ref), ref = 1; end
    validateattributes(ref, {'numeric'}, ...
                       {'scalar', 'integer', 'positive', '<=', S.nsteps}, ...
                       mfilename, 'ref', 2);

    Kref = seq_K(S, ref);

    t_fac = tic;
    [L, D, perm, Sscale] = ldl(Kref, 'vector');
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
    ctx.nnzK1   = nnz(Kref);
    ctx.nnzL    = nnz(L);
    ctx.fill_ratio = nnz(L) / max(nnz(Kref), 1);

    ctx.cond_ref = NaN;                 % filled only if the gate below has to look

    % nC backsolves, paid once: Sel does not move, so neither does K_ref^{-1}Sel.
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

    % AFTER t_setup is stamped, deliberately.  D is block diagonal (1x1 and 2x2
    % pivots) so condest(D) is only ~12 ms at ntot = 15759, but t_setup is a
    % PUBLISHED cost -- the "one factorization plus nC backsolves" claim -- and a
    % diagnostic has no business inflating it by 3.4%.  It used to be computed
    % only inside the failure branch, which left a healthy run unable to report
    % the conditioning its own gate is expressed in.
    ctx.rcond_D = 1 / local_condest(D);

    % The applier is the load-bearing piece of the cost argument, so its
    % correctness is checked here rather than only in the test suite: a wrong
    % permutation or scaling would still produce plausible-looking numbers.
    %
    % THE THRESHOLD MUST SCALE WITH THE REFERENCE.  relres is bounded below by
    % cond(K_ref)*eps no matter how right the applier is, so a FIXED 1e-8 is in
    % truth the assertion cond(K_ref) < 4.5e7 -- and it reports anything worse as
    % "the ldl convention has changed", which is a misdiagnosis that costs a
    % debugging session.  Measured on the rotor: cond(K_ref) 2.5e12 gives
    % relres 9.3e-6 with a perfectly correct applier.  So: refuse a SINGULAR
    % reference first, then compare relres against what its conditioning forces.
    % A wrong permutation gives relres >> 1 at ANY conditioning (measured: 20 to
    % 790), which is why the tolerance is capped at 1 and the wiring check
    % survives the loosening.
    relres = norm(Kref * YSel - full(S.Sel), 'fro') / ...
             max(norm(full(S.Sel), 'fro'), eps);
    if ~(relres < 1e-8)
        % Nothing below runs on the fast path, so condest(K_ref) -- an extra
        % sparse LU -- is free where it matters and t_setup is unchanged.
        n_bad = nnz(~isfinite(YSel));
        kappa = local_condest(Kref);
        ctx.cond_ref = kappa;

        if n_bad > 0 || ~isfinite(kappa) || 1/kappa < eps
            error('woodbury_context_init:singularReference', ...
                  ['K_%d (the frozen reference) is numerically singular ' ...
                   '(condest = %.3e, 1/condest(D) = %.3e, %d non-finite entries ' ...
                   'in K_%d^{-1}Sel), so there is nothing here for the applier to ' ...
                   'be right about.  K = S P L D L'' P'' S with L unit lower ' ...
                   'triangular and P, S invertible, so K is singular exactly when ' ...
                   'D is -- the fault is in the OPERATOR, not in the ldl ' ...
                   'convention and not in this file.  The usual cause is a ' ...
                   'rank-deficient coupling block C; build_stokes_sequence''s ' ...
                   'assert_coupling_feasible check names it at assembly time when ' ...
                   'the row count alone reveals it.'], ...
                  ref, kappa, ctx.rcond_D, n_bad, ref);
        end

        tol = min(1, max(1e-8, 50 * kappa * eps));
        if ~(relres < tol)
            error('woodbury_context_init:badApply', ...
                  ['woodbury_apply_ref does not invert K_%d ' ...
                   '(||K Y - Sel||/||Sel|| = %.3e, against %.3e allowed by ' ...
                   'condest(K) = %.3e).  Ill conditioning alone cannot produce ' ...
                   'this much -- the ldl permutation/scaling convention has ' ...
                   'changed.'], ref, relres, tol, kappa);
        end
    end
    ctx.apply_relres = relres;
end

%==========================================================================
function k = local_condest(A)
%LOCAL_CONDEST  condest without the two side effects it has on a caller.
%
%   condest estimates the 1-norm of A^{-1} with normest1, which DRAWS FROM THE
%   GLOBAL RANDOM STREAM.  Calling it here therefore advanced the RNG for
%   everything downstream of a context init, which is not a cost this function is
%   entitled to impose on its caller: it silently broke test_capacitance T9a and
%   test_recursive_growth, both of which seed with rng(0) and then draw.  The
%   state is saved and restored, and the expected singular-matrix warnings are
%   suppressed only around the call.
    st = rng;
    ws = warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    k = condest(A);
    warning(ws);
    rng(st);
end
