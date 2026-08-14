function solvers = varvisc_define_solver_list(params)
%VARVISC_DEFINE_SOLVER_LIST  MINRES solver/preconditioner registry for the
% variable-viscosity Stokes-immersed-rotor benchmark.
%
%   solvers = varvisc_define_solver_list(params)
%
%   Same registry mechanism as stokes_immersed_rotor/define_solver_list, but
%   measured against a sequence whose per-step update is numerically FULL-RANK:
%   the moving high-contrast viscosity field changes every nonzero of the fluid
%   block every step, so nothing here can lean on the parent benchmark's
%   K_n - K_1 = rank-2nC structure.  The Krylov-recycling arm of the parent
%   registry is intentionally absent.
%
%   Returns a cell array of solver structs.  Each per-step KKT system is
%   SYMMETRIC INDEFINITE and is solved with MINRES -- except the one GMRES arm
%   below, whose preconditioner is indefinite by construction; a solver entry
%   differs only in the (SPD) preconditioner it applies, or supplies its own
%   self-contained solve.  Adding a preconditioner is a single struct appended
%   here — the engine (solve_stokes_varvisc) and the driver pick it up
%   automatically for CSV columns, plots and the summary table.
%
%   Solver-struct fields:
%     .key    short id used for CSV column names and output filenames
%             (must be a valid MATLAB field name), e.g. 'minres_unprec'.
%     .label  display name for plot legends/titles.
%     .build  @(pc) -> Papply   preconditioner-apply handle passed as MINRES's
%             5th argument, or [] for the unpreconditioned solve.
%     .solve  @(K,b,tol,mit,pc) -> [x,flag,relres,iters]   OPTIONAL.  When
%             present (and non-empty) the engine calls it instead of the
%             build+minres path.  iters MUST be a SCALAR -- the engine assigns
%             it into one element of a per-step array, so a 1x2 [outer inner]
%             from gmres has to be collapsed by the closure.
%
%   pc is a context struct the engine fills (see solve_stokes_varvisc; unlike
%   the constant-viscosity engine there are no constant preconditioner pieces):
%     pc.Au_bc CURRENT-step BC-eliminated velocity block (changes every step)
%     pc.dP    nu-weighted lumped pressure-mass diagonal (apply: yp = rp ./ dP)
%     pc.nu_e  current element viscosities
%     pc.nU/nP velocity / pressure DOF counts
%     pc.nC    number of coupling rows at the current step (per step)
%     pc.K     the current-step KKT matrix (per step)
%     pc.step  the current step index n (per step)
%     pc.cache containers.Map handle for caching/refreshing factorizations
%
%   Refresh cadences (params.*_PREC_REFRESH; one independent knob per
%   preconditioner component, default Inf = build once):
%     BLOCKJAC_PREC_REFRESH  block-Jacobi ichol factor (the refreshed arm;
%                            the frozen arm has its own hard-wired Inf)
%     ILDL_PREC_REFRESH      incomplete-LDL factor C
%     DEFLAT_PREC_REFRESH    deflation subspace V
%     DINVERSE_PREC_REFRESH  exact A^-1 factor (sketched V methods).  If you
%                            ever set DEFLAT_PREC_REFRESH finite, set this to
%                            match -- otherwise the inverse-power sketch mixes
%                            a stale decomposition(K_1) with the current C.
%     EXACT_PREC_REFRESH     EXACT LDL factor C (the frozen exact_ldl_frozen arm)
%     ESKETCH_REF_REFRESH    frozen reference split factor C_ref of the E-sketch
%                            (Inf = once)
%
%   Two-level / deflation method knobs (params.*):
%     DEFLAT_SM_EIG        # smallest-|lambda| deflation vectors (report sm_eig)
%     DEFLAT_LG_EIG        # largest-|lambda|  deflation vectors (report lg_eig)
%     DEFLAT_Q             power-iteration rounds, shared by EVERY randomized
%                          sketch (gaussian/sjlt V and the low-rank D-sketch)
%     SKETCH_OVERSAMPLE    GLOBAL oversampling factor: every randomized sketch
%                          draws SKETCH_OVERSAMPLE * DEFLAT_SM_EIG columns and
%                          KEEPS them all -- the basis is only orthonormalized
%                          at the end, never truncated, so sketching is
%                          parameter-identical across the randomized methods.
%                          ('exact'/'polynomial' V are deterministic and keep
%                          dimension DEFLAT_SM_EIG.)
%     DEFLAT_TAU           deflation coarse-correction weight tau
%     DEFLAT_CHEB_DEGREE   Chebyshev degree (polynomial V; exact eigs band)
%     ILDL_MODE            incomplete-LDL pattern: 'nofill' | 'droptol'
%     ILDL_DROPTOL         drop tolerance when ILDL_MODE = 'droptol'
%     GMRES_MAXIT          iteration cap for the (unrestarted) gmres arm
%
%   COORDINATES.  Every basis cached across steps is held in PHYSICAL
%   coordinates and re-expressed in the current step's split coordinates on
%   use.  The ILDL factor C is refreshed every step (ILDL_PREC_REFRESH = 1)
%   while the bases are not (DEFLAT_PREC_REFRESH = Inf), and a basis in hat
%   coordinates is only a REPRESENTATION of a physical subspace: freezing the
%   representation across a refresh silently deflates a different subspace
%   each step.  See cached_basis for the mechanism.
%
%   Per the project convention this registry stays LOCAL to the benchmark;
%   the +src engine is kept preconditioner-agnostic.

    if nargin < 1 || isempty(params), params = struct(); end

    % --- coordinate-transport helpers ----------------------------------------
    % transport_V / ildl_coordinate_map / orth_trunc live in the subspace_recycle
    % study kernel and stay LOCAL (not promoted to +src).
    add_transport_path();

    % --- per-preconditioner refresh cadences (Inf = build once) ---------------
    R_blkjac = getdef(params, 'BLOCKJAC_PREC_REFRESH', 1);
    R_ildl   = getdef(params, 'ILDL_PREC_REFRESH',     1);
    R_deflat = getdef(params, 'DEFLAT_PREC_REFRESH',   Inf);
    R_dinv   = getdef(params, 'DINVERSE_PREC_REFRESH', Inf);
    R_exact  = getdef(params, 'EXACT_PREC_REFRESH',    Inf);
    % Own cadence, deliberately NOT shared with EXACT_PREC_REFRESH: that knob owns
    % an ILDL-struct factor used as a SMOOTHER, this one owns the exact split
    % factor of the REFERENCE system the E-sketch differences against.
    R_esk    = getdef(params, 'ESKETCH_REF_REFRESH',   Inf);

    % --- GMRES budget (full/unrestarted, see gmres_frozen_solve) --------------
    GMRES_MAXIT = getdef(params, 'GMRES_MAXIT', 300);

    % --- two-level / deflation hyperparameters --------------------------------
    % ONE sketch configuration for every randomized method: width
    % oversample*sm_eig, q power rounds, orthonormalize-only at the end.
    DEFL = struct( ...
        'sm_eig',       getdef(params, 'DEFLAT_SM_EIG',      500), ...
        'lg_eig',       getdef(params, 'DEFLAT_LG_EIG',      0), ...
        'q',            getdef(params, 'DEFLAT_Q',           2), ...
        'oversample',   getdef(params, 'SKETCH_OVERSAMPLE',  2), ...
        'tau',          getdef(params, 'DEFLAT_TAU',         0.5), ...
        'ildl_mode',    getdef(params, 'ILDL_MODE',          'nofill'), ...
        'droptol',      getdef(params, 'ILDL_DROPTOL',       1e-3), ...
        'cheb_degree',  getdef(params, 'DEFLAT_CHEB_DEGREE', 12));

    % --- symmetric E = C_1^-1 dK C_1^-T sketch (two_level_esketch) ------------
    % Same unified sketch parameters as the deflation arms; V keeps all
    % oversample*sm_eig columns (truncated only at numerical rank).  In this
    % benchmark dK = K_n - K_1 is numerically full-rank, so unlike the parent
    % there is no 2*nC ceiling on what the sketch can capture -- and no
    % guarantee that k columns contain every direction the operator moved.
    ESK = struct( ...
        'sm_eig',     DEFL.sm_eig, ...
        'oversample', DEFL.oversample, ...
        'q',          DEFL.q, ...
        'reorth',     true, ...
        'tau',        DEFL.tau, ...
        'ildl_mode',  DEFL.ildl_mode, ...
        'droptol',    DEFL.droptol);
    ESKETCH_K = round(ESK.oversample * ESK.sm_eig);

    solvers = {};

    solvers{end+1} = struct( ...
        'key',   'minres_unprec', ...
        'label', 'MINRES (unpreconditioned)', ...
        'build', @(pc) []);

    % SPD block-diagonal ("block Jacobi") at the CURRENT viscosity state; the
    % ichol factor is refreshed on its own cadence (default 1: rebuilt every
    % step, because the fluid block moves every step).
    solvers{end+1} = struct( ...
        'key',   'block_jacobi', ...
        'label', 'MINRES (block Jacobi, refreshed)', ...
        'build', @(pc) blockjac_build(pc, R_blkjac, 'blockjac_Lc', false));

    % The same block preconditioner FROZEN at step 1 (both the ichol factor and
    % the nu-weighted pressure diagonal).  The frozen-vs-refreshed gap is the
    % source benchmark's headline diagnostic: it vanishes on the static-nu
    % control case and opens as the viscosity field moves.
    solvers{end+1} = struct( ...
        'key',   'block_jacobi_frozen', ...
        'label', 'MINRES (block Jacobi, frozen at step 1)', ...
        'build', @(pc) blockjac_build(pc, Inf, 'blockjac_frozen', true));

    % Incomplete-LDL only (split solve, no coarse correction).
    solvers{end+1} = struct( ...
        'key',   'ildl_nofill', ...
        'label', 'MINRES (incomplete-LDL, no-fill)', ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 tl_solve(K, b, tol, mit, pc, 'none', DEFL, R_ildl, R_deflat, R_dinv));

    % Exact LDL of the step-1 KKT, FROZEN for every later step (own cadence).
    % Not an approximation: with nothing dropped, C = S^-1 P^T L |D|^{1/2} gives
    % M = C C^T = |K| and Ahat = C^-1 K C^-T = sign(D) -- 2 MINRES iterations on
    % the matrix it was built from.  Applied unchanged to K_n it isolates ONE
    % variable: how fast an EXACT factor stops preconditioning under the
    % FULL-RANK drift of the fluid block (in the parent benchmark the drift was
    % rank-2nC and this curve was a floor; here it is expected to degrade).
    solvers{end+1} = struct( ...
        'key',   'exact_ldl_frozen', ...
        'label', 'MINRES (exact LDL of step 1, frozen)', ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 exact_ldl_solve(K, b, tol, mit, pc, R_exact, DEFL.tau));

    % Two-level deflation (ILDL smoother + indefinite deflation, B = L^-T P L^-1),
    % one entry per V-building operation.
    for m = {'sjlt', 'gaussian', 'polynomial', 'exact'}
        meth = m{1};
        key  = ['two_level_' meth];
        opts = DEFL;  opts.method = meth;
        solvers{end+1} = struct( ...
            'key',   key, ...
            'label', sprintf('MINRES (ILDL + deflation L^{-T}PL^{-1}, %s V)', meth), ...
            'build', [], ...
            'solve', @(K,b,tol,mit,pc) ...
                     tl_solve(K, b, tol, mit, pc, meth, opts, R_ildl, R_deflat, R_dinv)); %#ok<AGROW>
    end

    % GMRES on the EXACT SIGNED inverse of the step-1 KKT, frozen (own cadence).
    % The one arm that is not MINRES, because it cannot be: the preconditioner is
    % K_1^-1 itself, which is INDEFINITE.  In the parent benchmark
    % K_1^-1 K_n = I + rank-2nC gave a finite-termination bound of 2nC+1
    % iterations; here the update is full-rank, so NO such bound exists and this
    % arm measures how the frozen exact inverse degrades without it (hitting
    % GMRES_MAXIT on the stress case is the expected negative result).
    solvers{end+1} = struct( ...
        'key',   'gmres_exact_inv_frozen', ...
        'label', 'GMRES (exact K_1^{-1} of step 1, frozen)', ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 gmres_frozen_solve(K, b, tol, mit, pc, R_exact, GMRES_MAXIT));

    % Two-level deflation whose coarse space is a randomized eigen-sketch of the
    % SYMMETRIC update operator E = C_1^-1 (K_n - K_1) C_1^-T, C_1 the exact
    % step-1 split factor -- the frozen step-1 factorization recycled into a
    % coarse space for the CURRENT system rather than into a solve.  E is the
    % operator that actually perturbs the split system MINRES runs on
    % (Ahat_1 = sign(D) + E), so its dominant EIGENVECTORS -- which the
    % similarity map C_n^T C_1^-T transports exactly, unlike the singular
    % spaces of the predecessor D = K_1^-1 dK sketch -- are the directions to
    % deflate.  Same smoother, same P^{1/2} coarse correction and same tau as
    % the rest of the family, so the comparison against two_level_gaussian /
    % two_level_exact isolates the ONE thing that differs: where V comes from.
    % What is reused is the FACTORIZATION of K_1, not a subspace: V depends on
    % dK and is rebuilt every step.  Registered LAST -> featured in
    % accuracy.png and the engine's progress print.
    solvers{end+1} = struct( ...
        'key',   'two_level_esketch', ...
        'label', sprintf(['MINRES (ILDL + deflation, C_1^{-1}(K_n-K_1)C_1^{-T} ' ...
                          'sketch V, k=%d, q=%d)'], ESKETCH_K, ESK.q), ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 tl_solve_esketch(K, b, tol, mit, pc, ESK, R_ildl, R_esk));

    solvers = solvers(:);
end

%==========================================================================
%  Solve / build closures
%==========================================================================
function Papply = blockjac_build(pc, refresh, key, freeze_dP)
%BLOCKJAC_BUILD  Block-Jacobi apply at the current (or frozen) viscosity state.
% The velocity ichol factor is cached under KEY and rebuilt on its own cadence;
% the nu-weighted pressure diagonal pc.dP is cheap and captured fresh every step
% UNLESS freeze_dP is set (the frozen arm caches both pieces at step 1).  The
% nC-dependent multiplier block is captured fresh every step either way.
    if freeze_dP
        pcs = cached(pc, key, refresh, ...
                     @() struct('Lc', ichol_robust(pc.Au_bc), 'dP', pc.dP));
        Lc = pcs.Lc;  dP = pcs.dP;
    else
        Lc = cached(pc, key, refresh, @() ichol_robust(pc.Au_bc));
        dP = pc.dP;
    end
    Papply = @(r) block_precond(r, pc.nU, pc.nP, pc.nC, Lc, dP);
end

function [x, fl, rr, it] = tl_solve(K, b, tol, mit, pc, method, opts, R_ildl, R_deflat, R_dinv)
%TL_SOLVE  Split two-level solve.  Builds (and refreshes, per-component) the ILDL
% factor, the optional exact-inverse factor and the coarse basis V, then runs the
% split-operator MINRES via src.precond.two_level_split_solve.
    [P, V] = two_level_parts(K, pc, method, opts, R_ildl, R_deflat, R_dinv);
    [x, fl, rr, it] = src.precond.two_level_split_solve(K, b, tol, mit, P, V, opts.tau);
end

function [x, fl, rr, it] = exact_ldl_solve(K, b, tol, mit, pc, refresh, tau)
%EXACT_LDL_SOLVE  Split MINRES preconditioned by the EXACT LDL^T factor of the KKT
% matrix from the last refresh step (default: step 1, then frozen forever).
%
% The same split solve as the ildl_nofill entry -- MINRES on Ahat = C^-1 K C^-T,
% no coarse space -- but C comes from make_ildl_precond's 'exact' mode, so M = C C^T
% is |K| exactly and Ahat = sign(D).  The fresh step therefore costs 2 iterations
% and every later step's count is pure drift of the operator.  tau is inert here
% (two_level_split_solve ignores it when V is empty).
%
% CACHE KEY.  'exact_ldl_frozen' is deliberately NOT of the form ['ildl_' mode] that
% two_level_parts builds, so it can never alias the smoother the ILDL / two-level
% entries share -- not even when a caller sets ILDL_MODE = 'exact'.  The two factors
% then coexist under different cadences, which is the whole point of this arm.
%
% VALIDITY PREDICATE.  Required, not decorative.  A frozen factor is APPLIED at
% every later step, so it can be handed a K of a different size: size(K,1) =
% nU + nP + nC, and nC drops if a Lagrange point leaves the fluid mesh.  Without
% the predicate this is an opaque dimension error; with it, `cached` rebuilds and
% warns -- and the warning matters, because a forced rebuild silently un-freezes
% the arm and changes what it measures.
    P = cached(pc, 'exact_ldl_frozen', refresh, ...
               @() src.precond.make_ildl_precond(K, struct('mode', 'exact')), ...
               @(v) numel(v.s) == size(K, 1));
    [x, fl, rr, it] = src.precond.two_level_split_solve(K, b, tol, mit, P, [], tau);
end

function [x, fl, rr, it] = gmres_frozen_solve(K, b, tol, mit, pc, refresh, gmaxit)
%GMRES_FROZEN_SOLVE  Unrestarted GMRES left-preconditioned by the EXACT SIGNED
% inverse of the KKT matrix from the last refresh step (default: step 1, frozen).
%
% K_1^-1 K_n = I + K_1^-1 (K_n - K_1).  In the parent (constant-viscosity)
% benchmark the update is rank <= 2nC and full GMRES terminates in <= 2nC+1
% iterations; HERE the update is numerically full-rank, so no finite-termination
% bound exists and the arm measures how the frozen indefinite inverse degrades.
%
% WHY GMRES.  MINRES needs an SPD preconditioner, so exact_ldl_frozen must replace
% D by |D|; GMRES has no such constraint and uses K_1^-1 verbatim, indefinite as
% it is.
%
% NO RESTART.  restart = [] is load-bearing: a restarted GMRES throws away the
% Krylov space.  gmaxit (params.GMRES_MAXIT) is a plain budget cap; hitting it on
% the moving-viscosity cases is a finding, not a bug.
%
% ITERATION COUNT.  MATLAB's gmres returns iter as a 1x2 [outer inner] even when
% unrestarted, but the engine assigns the 4th output into ONE element of a
% per-step array, so it is collapsed here (unrestarted -> outer == 1, so
% iter(end) is the total count).
%
% RELRES is the PRECONDITIONED relative residual, which is what left
% preconditioning measures.  The unambiguous accuracy check is Astat.solver_err
% (against the backslash solution), recorded by the engine anyway.
%
% CACHE KEY / VALIDITY.  Own key on the EXACT_PREC_REFRESH cadence, deliberately
% not 'dinv' and not exact_ldl_frozen's key -- same non-aliasing reasoning as
% exact_ldl_solve.  The predicate reads v.MatrixSize, NOT size(v): a decomposition
% is a scalar object, so size() reports [1 1] and the test would fire on every
% step, un-freezing the arm.  (K+K')/2 and an explicit 'ldl' keep a round-off
% asymmetry from silently downgrading the factorization to a general LU.
    dec = cached(pc, 'gmres_frozen_dec', refresh, ...
                 @() decomposition((K + K')/2, 'ldl'), ...
                 @(v) v.MatrixSize(1) == size(K, 1));
    [x, fl, rr, iter] = gmres(K, b, [], tol, min(gmaxit, mit), @(v) dec \ v);
    it = iter(end);
end

function [x, fl, rr, it] = tl_solve_esketch(K, b, tol, mit, pc, opts, R_ildl, R_ref)
%TL_SOLVE_ESKETCH  Split two-level solve whose coarse space is a randomized
% eigen-sketch of the symmetric update operator E = C_ref^-1 (K_n - K_ref) C_ref^-T.
% See varvisc_build_Esketch_V for the method; everything downstream of V is the
% ordinary src.precond.two_level_split_solve the rest of the family uses.
%
% WHAT IS REUSED IS THE FACTORIZATION, NOT THE SUBSPACE.  Every other basis in
% this file is cached across steps and transported into the current coordinates
% (cached_basis).  Here V depends on dK = K_n - K_ref, which changes every step,
% so there is nothing to freeze: the frozen object is the reference split factor,
% and V is rebuilt from it each step.
%
% The ILDL smoother comes from the SAME cache key the other two-level entries use,
% so adding this arm costs no extra smoother -- only its own coarse space.
%
% VALIDITY PREDICATE on the reference context: required for the same reason
% exact_ldl_frozen needs one (size(K,1) changes if a Lagrange point leaves the
% fluid mesh).
%
% info is stashed under 'esketch_info' (a plain cache set, not the refresh-cadence
% path) so the EFFECTIVE coarse dimension -- info.ncols, which rank truncation can
% put below k -- is observable after a run.
    P   = cached_ildl(K, pc, opts, R_ildl);
    ctx = cached(pc, 'esketch_ref', R_ref, ...
                 @() varvisc_esketch_ref_context(K), ...
                 @(v) v.n == size(K, 1));

    o = struct('k',      round(opts.oversample * opts.sm_eig), ...
               'q',      opts.q, ...
               'reorth', opts.reorth, ...
               'Cn',     current_C(pc, P));
    [V, info] = varvisc_build_Esketch_V(ctx, K, P, o);
    pc.cache('esketch_info') = struct('step', pc.step, 'val', info);

    [x, fl, rr, it] = src.precond.two_level_split_solve(K, b, tol, mit, P, V, opts.tau);
end

function P = cached_ildl(K, pc, opts, R_ildl)
%CACHED_ILDL  The ILDL smoother on its own refresh cadence, under the key every
% two-level entry shares (so it is built at most once per refresh-step no matter
% how many solvers ask for it).
    ikey = ['ildl_' opts.ildl_mode];
    ildl_opts = struct('mode', opts.ildl_mode);
    if strcmp(opts.ildl_mode, 'droptol'), ildl_opts.droptol = opts.droptol; end
    P = cached(pc, ikey, R_ildl, ...
               @() src.precond.make_ildl_precond(K, ildl_opts));
end

function [P, V] = two_level_parts(K, pc, method, opts, R_ildl, R_deflat, R_dinv)
%TWO_LEVEL_PARTS  The cached ILDL factor and coarse basis for one V-building
% method.  Shared by every two-level entry, so the ildl / dinv / V_<method> keys
% are built at most once per refresh-step no matter how many solvers ask.
% opts.oversample flows into build_deflation_V, which widens the gaussian/sjlt
% start block and keeps every column (orthonormalize-only, no truncation).
    P = cached_ildl(K, pc, opts, R_ildl);

    if strcmp(method, 'none')
        V = [];
        return;
    end
    dA = [];
    % Only the smallest-mode inverse-power sketch needs A^-1; the large-only
    % ablation (sm_eig=0) sketches the forward operator Ahat, so skip it then.
    if any(strcmp(method, {'gaussian', 'sjlt'})) && opts.sm_eig >= 1
        dA = cached(pc, 'dinv', R_dinv, @() decomposition(K));
    end
    o = opts;  o.method = method;
    V = cached_basis(pc, ['V_' method], R_deflat, K, P, ...
                     @() src.precond.build_deflation_V(K, P, o, dA));
end

function V = cached_basis(pc, key, refresh, K, P, buildFn)
%CACHED_BASIS  Refresh cache for a DEFLATION BASIS, held in PHYSICAL coordinates.
%
% Same cadence as `cached`, but what is stored is the PHYSICAL basis U = C^-T V
% rather than the hat-coordinate V that build_deflation_V returns.  MINRES runs on
% the SPLIT operator Ahat_n = C_n^-1 K_n C_n^-T with yhat = C_n^T x, so a basis
% expressed in hat coordinates denotes the physical subspace C_n^-T span(V).  With
% ILDL_PREC_REFRESH = 1 the factor C is rebuilt from scratch every step -- ldl
% re-derives the fill-reducing permutation p, the scaling S and the 1x1/2x2 pivot
% structure of D from K_n -- so a V cached in hat coordinates silently denotes a
% DIFFERENT physical subspace at every later step.  Storing U and mapping it
% forward as V_n = orth(C_n^T U) preserves the physical span EXACTLY, because
% C_n^-T (C_n^T U) = U.  Re-orthonormalizing changes the basis but not the span,
% and deflation only sees the span (deflation_Psqrt_apply does need V'V = I for
% its (I - VV') projector, which is what transport_V's orth_trunc supplies).
%
% Each entry carries TWO step stamps:
%   .step   step at which U was BUILT               -> drives the refresh cadence
%   .hstep  step whose hat coordinates .V is in     -> per-step transport memo
% two_level_parts is called once per two-level solver entry per step, so without
% .hstep the C^T multiply + QR would run several times per step.
%
% At a (re)build step nothing is transported: build_deflation_V already returns an
% orthonormal V in the CURRENT step's hat coordinates, so it is returned verbatim
% and we pay only one applyCtinv to stash U.
    c = pc.cache;
    n = size(K, 1);

    rebuild = true;
    if isKey(c, key)
        e = c(key);
        % size(K,1) changes when Lagrange points leave the fluid mesh; a physical
        % basis with a stale row count cannot be transported, so force a rebuild.
        rebuild = size(e.U, 1) ~= n || ...
                  ((e.step ~= pc.step) && (mod(pc.step - 1, refresh) == 0));
    end

    if rebuild
        V = buildFn();                                  % CURRENT-step hat coords
        c(key) = struct('step',  pc.step, 'U', P.applyCtinv(V), ...
                        'hstep', pc.step, 'V', V);
        return;
    end

    if e.hstep == pc.step
        V = e.V;                                        % memo hit within the step
        return;
    end

    [V, info] = transport_V(e.U, P, current_C(pc, P));   % orth(C_n^T U)
    if info.rank_drop > 0
        % A smaller but clean coarse space beats a rank-deficient one (E = V'Ahat^2 V
        % must stay SPD), but a silent shrink must not go unnoticed.
        warning('varvisc_define_solver_list:basisRankDrop', ...
                ['step %d: transporting %s dropped %d of %d columns ' ...
                 '(numerically dependent after the C^T map)'], ...
                pc.step, key, info.rank_drop, info.k_in);
    end
    e.hstep = pc.step;
    e.V     = V;
    c(key)  = e;
end

function C = current_C(pc, P)
%CURRENT_C  Explicit split factor C = S^-1 P^T L |D|^{1/2} for THIS step, memoized
% so the two-level entries share one materialization instead of rebuilding it each.
    c = pc.cache;
    if isKey(c, 'ildl_C')
        e = c('ildl_C');
        if e.step == pc.step && size(e.val, 1) == numel(P.s)
            C = e.val;
            return;
        end
    end
    C = ildl_coordinate_map(P);
    c('ildl_C') = struct('step', pc.step, 'val', C);
end

function add_transport_path()
%ADD_TRANSPORT_PATH  Put the subspace_recycle kernel on the MATLAB path.
% Holds transport_V, ildl_coordinate_map and orth_trunc -- LOCAL helpers, not
% promoted to +src.
    if ~isempty(which('transport_V')), return; end
    thisDir   = fileparts(mfilename('fullpath'));
    kernelDir = fullfile(fileparts(thisDir), 'linear_solves', ...
                         'subspace_recycle', 'kernel');
    if exist(kernelDir, 'dir')
        addpath(kernelDir);
    else
        error('varvisc_define_solver_list:noKernel', ...
              'coordinate-transport kernel not found at %s', kernelDir);
    end
end

function v = cached(pc, key, refresh, buildFn, isValidFn)
%CACHED  Per-component refresh cache keyed in pc.cache (one rebuild at most per
% step, on the mod(step-1,refresh)==0 cadence).  Shared keys (ildl/dinv) are thus
% built once per refresh-step and reused across solver entries within that step.
%
% ISVALIDFN (optional) is a predicate on the CACHED VALUE that must also hold
% before the value may be reused.  It exists for FROZEN factors (refresh = Inf),
% which are APPLIED at every later step and can therefore meet a K of a different
% size -- size(K,1) = nU + nP + nC changes when a Lagrange point leaves the fluid
% mesh, the same hazard cached_basis guards with size(e.U,1) ~= n.
    if nargin < 5, isValidFn = []; end
    c = pc.cache;
    if ~isKey(c, key)
        v = buildFn();
        c(key) = struct('step', pc.step, 'val', v);
        return;
    end
    e = c(key);
    stale = ~isempty(isValidFn) && ~isValidFn(e.val);
    if stale
        % A forced rebuild un-freezes the factor, so this step no longer measures
        % what the arm exists to measure.  Never let that happen silently.
        warning('varvisc_define_solver_list:cacheShapeChanged', ...
                ['step %d: cached "%s" no longer fits the current system (the ' ...
                 'multiplier count changed); rebuilding -- this step is NOT the ' ...
                 'frozen factor.'], pc.step, key);
    end
    if stale || (e.step ~= pc.step && mod(pc.step - 1, refresh) == 0)
        v = buildFn();
        c(key) = struct('step', pc.step, 'val', v);
    else
        v = e.val;
    end
end

function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

%==========================================================================
%  Preconditioner-apply helpers
%==========================================================================
function Lc = ichol_robust(A)
%ICHOL_ROBUST  ichol('nofill') with diagonal-compensation escalation; the 100:1
% viscosity contrast can break the plain factorization.
    try
        Lc = ichol(A, struct('type', 'nofill'));
        return;
    catch
    end
    alpha = 1e-3;
    for k = 1:8
        try
            Lc = ichol(A, struct('type', 'nofill', 'diagcomp', alpha));
            return;
        catch
            alpha = alpha * 10;
        end
    end
    Lc = ichol(A, struct('type', 'ict', 'droptol', 1e-3, 'diagcomp', 0.1));
end

function y = block_precond(r, nU, nP, nC, Lc, dP)
%BLOCK_PRECOND  Apply the SPD block-diagonal ("block Jacobi") preconditioner
% P^{-1} r for the variable-viscosity Stokes KKT system.
%   Pu   ~ Avel(nu_e)             (applied via ichol factor Lc)
%   Pp   ~ (1/nu)-weighted lumped pressure mass (apply: yp = rp ./ dP)
%   Plam = I
    ru = r(1:nU);
    rp = r(nU + (1:nP));
    rl = r(nU + nP + (1:nC));

    yu = Lc' \ (Lc \ ru);
    yp = rp ./ dP;
    yl = rl;

    y = [yu; yp; yl];
end
