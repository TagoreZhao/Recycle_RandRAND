function solvers = define_solver_list(params)
%DEFINE_SOLVER_LIST  MINRES solver/preconditioner registry for the
% Stokes-immersed-rotor benchmark (simplified deal.II step-70).
%
%   solvers = define_solver_list(params)
%
%   Returns a cell array of solver structs.  Each per-step KKT system is
%   SYMMETRIC INDEFINITE and is solved with MINRES -- except the one GMRES arm
%   below, whose preconditioner is indefinite by construction; a solver entry
%   differs only in the (SPD) preconditioner it applies, or supplies its own
%   self-contained solve.  This is the extensibility seam: adding a preconditioner
%   is a single
%   struct appended here — the engine (solve_stokes_immersed) and the driver
%   (run_benchmark, make_paper_summary_table) pick it up automatically for CSV
%   columns, plots and the summary table.
%
%   Solver-struct fields:
%     .key    short id used for CSV column names and output filenames
%             (must be a valid MATLAB field name), e.g. 'minres_unprec'.
%     .label  display name for plot legends/titles.
%     .build  @(pc) -> Papply   preconditioner-apply handle passed as MINRES's
%             5th argument, or [] for the unpreconditioned solve.
%     .solve  @(K,b,tol,mit,pc) -> [x,flag,relres,iters]   OPTIONAL.  When
%             present (and non-empty) the engine calls it instead of the
%             build+minres path; used by the two-level scheme, which runs MINRES
%             on the split operator C^-1 K C^-T and unwinds, and by the GMRES
%             arm (the build path hard-codes minres).  iters MUST be a SCALAR --
%             the engine assigns it into one element of a per-step array, so a
%             1x2 [outer inner] from gmres has to be collapsed by the closure.
%
%   pc is a context struct the engine fills:
%     pc.Lc    ichol factor of the (BC-eliminated) velocity block Avel
%     pc.Rp    chol  factor of the pressure mass matrix Dp
%     pc.Au_bc the (constant) BC-eliminated velocity block (ichol source)
%     pc.nu    kinematic viscosity
%     pc.nU/nP velocity / pressure DOF counts
%     pc.nC    number of coupling rows at the current step (per step)
%     pc.K     the current-step KKT matrix (per step)
%     pc.step  the current step index n (per step)
%     pc.cache containers.Map handle for caching/refreshing factorizations
%
%   Refresh cadences (params.*_PREC_REFRESH; mirror report/solve_deflate_M_P,
%   one independent knob per preconditioner component, default Inf = build once):
%     BLOCKJAC_PREC_REFRESH  block-Jacobi ichol factor
%     ILDL_PREC_REFRESH      incomplete-LDL factor C
%     DEFLAT_PREC_REFRESH    deflation subspace V
%     DINVERSE_PREC_REFRESH  exact A^-1 factor (sketched V methods)
%     EXACT_PREC_REFRESH     EXACT LDL factor C (the frozen exact_ldl_frozen arm)
%
%   Two-level / deflation method knobs (params.*; defaults mirror
%   report/ball_surface/run_benchmark.m):
%     DEFLAT_SM_EIG        # smallest-|lambda| deflation vectors (report sm_eig)
%     DEFLAT_LG_EIG        # largest-|lambda|  deflation vectors (report lg_eig)
%     DEFLAT_Q             sketch power-iteration rounds (gaussian/sjlt V)
%     DEFLAT_TAU           deflation coarse-correction weight tau
%     DEFLAT_CHEB_DEGREE   Chebyshev degree (polynomial V; exact eigs band)
%     DEFLAT_RECYCLE_K     # Krylov vectors recycled per step (two_level_krylov)
%     ILDL_MODE            incomplete-LDL pattern: 'nofill' | 'droptol'
%     ILDL_DROPTOL         drop tolerance when ILDL_MODE = 'droptol'
%     GMRES_MAXIT          iteration cap for the (unrestarted) gmres_exact_inv_frozen
%                          arm; must stay above 2*nC+1 or it caps the claim
%
%   Low-rank sketch knobs (the 'two_level_lowrank_sketch' entry):
%     LOWRANK_SM_EIG       # small eigendirections of interest
%     LOWRANK_OVERSAMPLE   oversampling FACTOR; the sketch width is
%                          k = LOWRANK_OVERSAMPLE * LOWRANK_SM_EIG
%     LOWRANK_SKETCH_Q     power-iteration rounds q
%     LOWRANK_REF_REFRESH  cadence of the frozen reference factorization (Inf = once)
%
%   Low-rank finite termination (the 'gmres_exact_inv_frozen' entry).  K_n - K_1 is
%   symmetric of rank 2*rank(dC) <= 2*nC, so left-preconditioning K_n with the EXACT
%   SIGNED inverse of K_1 gives I plus a rank-r update and unrestarted GMRES must
%   terminate in <= r+1 iterations.  MINRES cannot run this operator (K_1^-1 is
%   indefinite), which is exactly why the arm is GMRES.  See gmres_frozen_solve and
%   write_lowrank_bound_figure.
%
%   Low-rank A^-1 B sketch (the 'two_level_lowrank_sketch' entry).  The same two-level
%   scheme as the rest of the family -- ILDL smoother, P^{1/2} coarse correction on
%   Ahat^2, MINRES on the split operator -- differing ONLY in how the coarse space is
%   built.  With A_1 = K_1 factored once and frozen and A_2 = K_n the current system,
%   the directions the update moves are the range of the NONSYMMETRIC D = A_1^-1(A_2 - A_1),
%   whose dominant left singular subspace is taken by randomized power iteration:
%       Y = (D D')^q D Omega,   Omega = randn(n, k),   V = orth(C_n' Y).
%   Because K_n - K_1 = U B U' with U = [dC, Sel] and B invertible, range(D) is exactly
%   K_1^-1 range(U), of dimension <= 2*nC -- so at k >= 2*nC this coarse space contains
%   every direction the operator update can have moved, at <= 2*nC EFFECTIVE columns
%   rather than DEFLAT_SM_EIG of them, and with no refactorization after step 1.  Below
%   that rank the sketch is not merely weaker but WORSE than no coarse space at all --
%   see the measurements at the LOWR defaults below.  What is recycled
%   here is the FACTORIZATION of A_1, not the subspace: V depends on dK = K_n - K_1 and
%   is rebuilt every step, which is why this entry does not use cached_basis.  See
%   build_lowrank_sketch_V.
%
%   Krylov recycling (the 'two_level_krylov' entry).  Consecutive KKT systems differ
%   only through the moving coupling block C(t_n), so the directions MINRES converged
%   slowly on at step n-1 are the ones it will converge slowly on at step n.  Because
%   MINRES runs on the SPLIT operator, the vector it hands to its preconditioner each
%   iteration already IS the ILDL-preconditioned residual, so make_recording_pdef
%   captures the last DEFLAT_RECYCLE_K of them as a free side effect of the ordinary
%   minres call (no separate Lanczos, no extra matvec, iteration count unchanged).
%   The block is carried in pc.cache — in PHYSICAL coordinates, see cached_basis —
%   and, at the next step, mapped into that step's hat coordinates and appended to
%   the SAME Gaussian coarse space two_level_gaussian uses (augment_recycle_V), so
%   the pair is a controlled comparison, identical at step 1 where nothing is
%   recycled yet.  Ported from the SPD/PCG scheme in
%   Preconditioner_Recycle/report/ball_surface_krylov_recycle.
%
%   COORDINATES.  Every basis cached across steps — the coarse space V_<method> and
%   the recycled block alike — is held in PHYSICAL coordinates and re-expressed in
%   the current step's split coordinates on use.  The ILDL factor C is refreshed
%   every step (ILDL_PREC_REFRESH = 1) while the bases are not (DEFLAT_PREC_REFRESH
%   = Inf), and a basis in hat coordinates is only a REPRESENTATION of a physical
%   subspace: freezing the representation across a refresh silently deflates a
%   different subspace each step.  See cached_basis for the mechanism and the fix.
%
%   Mirrors define_motion_list (the geometry/motion registry).  Per the project
%   convention, this preconditioner registry stays LOCAL to the benchmark (it
%   changes/grows and is not geometry-persistent); the +src engine is kept
%   preconditioner-agnostic.

    if nargin < 1 || isempty(params), params = struct(); end

    % --- coordinate-transport helpers ----------------------------------------
    % transport_V / ildl_coordinate_map / orth_trunc live in the subspace_recycle
    % study kernel and stay LOCAL (not promoted to +src) until the production
    % numbers confirm the approach.  Bootstrapping the path here rather than in
    % run_benchmark means every caller -- the benchmark, profile_components, the
    % tests -- picks them up automatically and cannot forget.
    add_transport_path();

    % --- per-preconditioner refresh cadences (report-style; Inf = build once) ---
    R_blkjac = getdef(params, 'BLOCKJAC_PREC_REFRESH', Inf);
    R_ildl   = getdef(params, 'ILDL_PREC_REFRESH',     1);
    R_deflat = getdef(params, 'DEFLAT_PREC_REFRESH',   Inf);
    R_dinv   = getdef(params, 'DINVERSE_PREC_REFRESH', Inf);
    R_exact  = getdef(params, 'EXACT_PREC_REFRESH',    Inf);
    % Own cadence, deliberately NOT shared with EXACT_PREC_REFRESH: that knob owns an
    % ILDL-struct factor used as a SMOOTHER, this one owns the raw ldl factors of the
    % REFERENCE system the low-rank sketch differences against.  Different objects,
    % so different knobs (the one-knob-per-component convention above).
    R_lowref = getdef(params, 'LOWRANK_REF_REFRESH',   Inf);

    % --- GMRES budget (the low-rank arm; full/unrestarted, see gmres_frozen_solve) ---
    GMRES_MAXIT = getdef(params, 'GMRES_MAXIT', 300);

    % --- two-level / deflation hyperparameters (from params, with defaults) ---
    %   Defaults mirror report/ball_surface/run_benchmark.m (sm_eig=500, tau=0.5,
    %   q=2, cheb_degree=12).  The polynomial reject band is set from EXACT eigs
    %   of (A,M), so there is no lam_cut_frac knob.
    DEFL = struct( ...
        'sm_eig',       getdef(params, 'DEFLAT_SM_EIG',       500), ...
        'lg_eig',       getdef(params, 'DEFLAT_LG_EIG',       0), ...
        'q',            getdef(params, 'DEFLAT_Q',            2), ...
        'tau',          getdef(params, 'DEFLAT_TAU',          0.5), ...
        'ildl_mode',    getdef(params, 'ILDL_MODE',           'nofill'), ...
        'droptol',      getdef(params, 'ILDL_DROPTOL',        1e-3), ...
        'cheb_degree',  getdef(params, 'DEFLAT_CHEB_DEGREE',  12), ...
        'recycle_k',    getdef(params, 'DEFLAT_RECYCLE_K',    50));

    % --- low-rank A^-1 B sketch hyperparameters (two_level_lowrank_sketch) ------
    % The sketch width is the oversampling FACTOR times the number of small
    % eigendirections of interest, and V keeps all k columns (Omega is n-by-k, V is
    % n-by-k; there is no post-hoc truncation back to sm_eig).  rank(D) <= 2*nC caps
    % what k can actually buy -- see build_lowrank_sketch_V.
    %
    % THE DEFAULT PUTS k ABOVE 2*nC ON PURPOSE.  At h0 = 0.05 nC is 48 for bar_rotating
    % and 120 for the disks, so 2*nC <= 240 and k = 2*125 = 250 clears every case.
    % Below that rank the sketch keeps only the directions the UPDATE moved most, which
    % is not the same as the directions the OPERATOR is worst conditioned in, and it
    % measures WORSE than no coarse space at all.  Measured, MINRES iterations:
    %   h0=0.1  bar  (2nC= 48) step 2: smoother alone 385 | k= 15  411 | k>=48  273
    %   h0=0.05 disk (2nC=240) step 3: smoother alone 1931 | k=100 2119 | k=250 1184
    % test_lowrank_sketch_V T9 pins that boundary, so lowering sm_eig below nC is an
    % experiment, not a saving.
    LOWR = struct( ...
        'sm_eig',     getdef(params, 'LOWRANK_SM_EIG',    125), ...
        'oversample', getdef(params, 'LOWRANK_OVERSAMPLE',  2), ...
        'q',          getdef(params, 'LOWRANK_SKETCH_Q',    2), ...
        'reorth',     true, ...
        'tau',        DEFL.tau, ...
        'ildl_mode',  DEFL.ildl_mode, ...
        'droptol',    DEFL.droptol);
    LOWRANK_K = LOWR.oversample * LOWR.sm_eig;

    solvers = {};

    solvers{end+1} = struct( ...
        'key',   'minres_unprec', ...
        'label', 'MINRES (unpreconditioned)', ...
        'build', @(pc) []);

    % SPD block-diagonal ("block Jacobi"); ichol factor refreshed on its own cadence.
    solvers{end+1} = struct( ...
        'key',   'block_jacobi', ...
        'label', 'MINRES (block Jacobi)', ...
        'build', @(pc) blockjac_build(pc, R_blkjac));

    % Incomplete-LDL only (split solve, no coarse correction).
    solvers{end+1} = struct( ...
        'key',   'ildl_nofill', ...
        'label', 'MINRES (incomplete-LDL, no-fill)', ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 tl_solve(K, b, tol, mit, pc, 'none', DEFL, R_ildl, R_deflat, R_dinv));

    % Exact LDL of the step-1 KKT, FROZEN for every later step (own cadence).
    % Not an approximation: with nothing dropped, C = S^-1 P^T L |D|^{1/2} gives
    % M = C C^T = |K| and Ahat = C^-1 K C^-T = sign(D), a matrix whose eigenvalues
    % are exactly +-1 -- 2 MINRES iterations on the matrix it was built from.
    % Applied unchanged to K_n it isolates ONE variable: how fast an EXACT factor
    % stops preconditioning as the coupling block C(t_n) drifts, with smoother
    % quality removed from the comparison.  K_n - K_1 is symmetric of rank <= 2*nC,
    % so Ahat_n is sign(D_1) plus a rank-<=2*nC update: this curve is the FLOOR the
    % deflation and Krylov-recycling arms are trying to reach cheaply, not a
    % competitor to them.  EXACT_PREC_REFRESH = Inf keeps the step-1 factor forever.
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
    % The one arm in the registry that is not MINRES, because it cannot be: the
    % preconditioner is K_1^-1 itself, which is INDEFINITE.  That is the point.
    % exact_ldl_frozen has to SPD-ify the same factor for MINRES (M = |K_1|), so
    % what MINRES sees is sign(D_1) + rank-2nC; GMRES sees
    %     K_1^-1 K_n = I + K_1^-1 (K_n - K_1),
    % an identity plus a rank-r update (r = 2 rank(dC) <= 2 nC), whose minimal
    % polynomial has degree <= r+1.  Unrestarted GMRES must therefore terminate in
    % at most 2 nC + 1 iterations -- the claim write_lowrank_bound_figure plots.
    % Registered BEFORE two_level_krylov so the last-solver privileges (accuracy.png,
    % the relres / solver_err_last CSV columns, the engine progress print) stay put.
    solvers{end+1} = struct( ...
        'key',   'gmres_exact_inv_frozen', ...
        'label', 'GMRES (exact K_1^{-1} of step 1, frozen; I + rank-2n_C)', ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 gmres_frozen_solve(K, b, tol, mit, pc, R_exact, GMRES_MAXIT));

    % Two-level deflation whose coarse space is a randomized sketch of the update
    % operator D = K_1^-1 (K_n - K_1) -- the frozen step-1 factorization recycled into
    % a coarse space for the CURRENT system rather than into a solve.  Same smoother,
    % same P^{1/2} coarse correction and same tau as the rest of the family, so the
    % comparison against two_level_gaussian / two_level_exact isolates the ONE thing
    % that differs: where V comes from.  Registered after gmres_exact_inv_frozen and
    % before two_level_krylov so that registry rows 1-9 keep their existing plot
    % styles and the last-solver privileges (accuracy.png, the relres /
    % solver_err_last CSV columns, the engine progress print) stay with krylov.
    solvers{end+1} = struct( ...
        'key',   'two_level_lowrank_sketch', ...
        'label', sprintf(['MINRES (ILDL + deflation, K_1^{-1}(K_n-K_1) sketch V, ' ...
                          'k=%d, q=%d)'], LOWRANK_K, LOWR.q), ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 tl_solve_lowrank(K, b, tol, mit, pc, LOWR, R_ildl, R_lowref));

    % Krylov recycling: the gaussian coarse space + the last DEFLAT_RECYCLE_K
    % ILDL-preconditioned residuals harvested from the PREVIOUS step's solve.
    % Registered LAST -> featured in accuracy.png and the engine's progress print.
    kopts = DEFL;  kopts.method = 'gaussian';
    solvers{end+1} = struct( ...
        'key',   'two_level_krylov', ...
        'label', sprintf('MINRES (ILDL + deflation, gaussian V + %d recycled Krylov)', ...
                         kopts.recycle_k), ...
        'build', [], ...
        'solve', @(K,b,tol,mit,pc) ...
                 tl_solve_krylov(K, b, tol, mit, pc, kopts, R_ildl, R_deflat, R_dinv));

    solvers = solvers(:);
end

%==========================================================================
%  Solve / build closures
%==========================================================================
function Papply = blockjac_build(pc, refresh)
%BLOCKJAC_BUILD  Block-Jacobi apply; the velocity ichol factor is cached and
% rebuilt on its own refresh cadence (default Inf -> built once per case).  The
% nC-dependent multiplier block is captured fresh every step.
    Lc = cached(pc, 'blockjac_Lc', refresh, ...
                @() ichol(pc.Au_bc, struct('type', 'nofill')));
    Papply = @(r) block_precond(r, pc.nU, pc.nP, pc.nC, Lc, pc.Rp, pc.nu);
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
% and every later step's count is pure drift of C(t_n).  tau is inert here
% (two_level_split_solve ignores it when V is empty) and is passed only so the
% signature matches the other split entries.
%
% CACHE KEY.  'exact_ldl_frozen' is deliberately NOT of the form ['ildl_' mode] that
% two_level_parts builds, so it can never alias the smoother the ILDL / two-level
% entries share -- not even when a caller sets ILDL_MODE = 'exact', which makes
% THEIR key 'ildl_exact'.  The two factors then coexist under different cadences
% (ILDL_PREC_REFRESH vs EXACT_PREC_REFRESH), which is the whole point of this arm.
%
% VALIDITY PREDICATE.  Required, not decorative.  A frozen factor is APPLIED at
% every later step, so unlike the every-step ILDL it can be handed a K of a
% different size: size(K,1) = nU + nP + nC, and nC drops if a Lagrange point leaves
% the fluid mesh (the hazard cached_basis guards with size(e.U,1) ~= n).  Without
% the predicate this is an opaque dimension error inside applyCinv's `s .* r`; with
% it, `cached` rebuilds and warns -- and the warning matters, because a forced
% rebuild silently un-freezes the arm and changes what it measures.
    P = cached(pc, 'exact_ldl_frozen', refresh, ...
               @() src.precond.make_ildl_precond(K, struct('mode', 'exact')), ...
               @(v) numel(v.s) == size(K, 1));
    [x, fl, rr, it] = src.precond.two_level_split_solve(K, b, tol, mit, P, [], tau);
end

function [x, fl, rr, it] = gmres_frozen_solve(K, b, tol, mit, pc, refresh, gmaxit)
%GMRES_FROZEN_SOLVE  Unrestarted GMRES left-preconditioned by the EXACT SIGNED
% inverse of the KKT matrix from the last refresh step (default: step 1, frozen).
%
% Tests the finite-termination property the benchmark's low-rank structure implies.
% K_n - K_1 is symmetric of rank 2 rank(dC) <= 2 nC, so
%     K_1^-1 K_n = I + K_1^-1 (K_n - K_1)
% is an identity plus a rank-r update; its minimal polynomial has degree <= r+1 and
% full GMRES annihilates the residual in at most that many iterations.  Nothing here
% is an approximation to be tuned -- the arm either meets 2 nC + 1 or it does not.
%
% WHY GMRES.  MINRES needs an SPD preconditioner, so exact_ldl_frozen must replace D
% by |D| and ends up on sign(D_1) + rank-r, which has 2 distinct eigenvalues rather
% than 1 and loses the clean I + low-rank statement.  GMRES has no such constraint
% and can use K_1^-1 verbatim, indefinite as it is.
%
% NO RESTART.  restart = [] is load-bearing: a restarted GMRES throws away the
% Krylov space that carries the finite-termination argument.  gmaxit
% (params.GMRES_MAXIT) is then a plain cap on total iterations; it must stay
% comfortably above 2 nC + 1 or the arm measures the budget instead of the claim.
%
% ITERATION COUNT.  MATLAB's gmres returns iter as a 1x2 [outer inner] even when
% unrestarted, but solve_stokes_immersed assigns the 4th output into ONE element of
% a per-step array.  Forwarding it verbatim is a run-time error, so it is collapsed
% here; unrestarted means outer == 1, hence iter(end) is the total count.
%
% RELRES is the PRECONDITIONED relative residual ||M^-1(b - Kx)|| / ||M^-1 b||, which
% is what left preconditioning measures.  The unambiguous accuracy check is
% Astat.solver_err (against the backslash solution), recorded by the engine anyway.
%
% CACHE KEY / VALIDITY.  Own key on the EXACT_PREC_REFRESH cadence, deliberately not
% 'dinv' (whose cadence is DINVERSE_PREC_REFRESH and which the sketched-V arms share)
% and not exact_ldl_frozen's key -- same non-aliasing reasoning as exact_ldl_solve.
% The predicate is required for the same reason: a frozen factor is APPLIED at every
% later step and size(K,1) = nU + nP + nC shrinks if a Lagrange point leaves the mesh.
% The predicate reads v.MatrixSize, NOT size(v): a decomposition is a scalar object,
% so size() reports [1 1] and the test would fire on every step, un-freezing the arm
% (loudly -- `cached` warns -- but every step, which is worse than useless).
% (K+K')/2 and an explicit 'ldl' keep a round-off asymmetry from silently downgrading
% the factorization to a general LU.
    dec = cached(pc, 'gmres_frozen_dec', refresh, ...
                 @() decomposition((K + K')/2, 'ldl'), ...
                 @(v) v.MatrixSize(1) == size(K, 1));
    [x, fl, rr, iter] = gmres(K, b, [], tol, min(gmaxit, mit), @(v) dec \ v);
    it = iter(end);
end

function [x, fl, rr, it] = tl_solve_krylov(K, b, tol, mit, pc, opts, R_ildl, R_deflat, R_dinv)
%TL_SOLVE_KRYLOV  Split two-level solve whose coarse space is the cached gaussian V
% AUGMENTED with the ILDL-preconditioned residuals recycled from the PREVIOUS step.
% Same recipe as src.precond.two_level_split_solve, inlined only because MINRES has
% to be handed a recording preconditioner instead of the bare coarse operator.
%
% The harvested block is refreshed every step (a plain pc.cache set, NOT the
% refresh-cadence `cached` path) and is held in PHYSICAL coordinates under the key
% `krylov_U`, then mapped into this step's split coordinates on use — same reason
% cached_basis stores the coarse space physically.  P and Vbase come from the SAME
% cache keys two_level_gaussian uses, so the base coarse space is literally the
% same object and the two solvers differ only by the recycled columns.
    [P, Vbase] = two_level_parts(K, pc, 'gaussian', opts, R_ildl, R_deflat, R_dinv);

    % --- attach the previous step's block, mapped from PHYSICAL coordinates into
    %     THIS step's hat coordinates.  It was harvested against C_{n-1}, and
    %     ILDL_PREC_REFRESH = 1 rebuilds C from scratch, so reusing it verbatim
    %     would recycle a DIFFERENT physical subspace (see cached_basis).  Guards
    %     unchanged: freshness AND dimension (nC, and hence size(K,1), can change
    %     when Lagrange points leave the fluid mesh).
    %     No QR here on purpose: augment_recycle_V unit-normalizes the columns,
    %     projects off Vbase and orth()s the remainder, so the raw C^T multiply is
    %     all W needs -- truncating first would only throw information away.
    W = zeros(size(K, 1), 0);
    if isKey(pc.cache, 'krylov_U')
        e = pc.cache('krylov_U');
        if e.step == pc.step - 1 && size(e.U, 1) == size(K, 1)
            W = current_C(pc, P)' * e.U;             % C_n^T U : physical -> hat
        end
    end
    V = augment_recycle_V(Vbase, W);   % Vbase orthonormal, from cached_basis

    Afun  = @(y) P.applyCinv(K * P.applyCtinv(y));   % Ahat = C^-1 K C^-T
    btil  = P.applyCinv(b);                          % C^-1 b
    Ahat2 = @(z) Afun(Afun(z));                      % Ahat^2 (SPD)
    Pdef  = src.precond.deflation_Psqrt_apply(V, Ahat2, opts.tau, 'handle');
    [Mfun, getU] = make_recording_pdef(Pdef, numel(btil), opts.recycle_k);

    [y, fl, rr, it] = minres(Afun, btil, tol, mit, Mfun);
    x = P.applyCtinv(y);                             % recover x = C^-T y

    % Store the harvest in PHYSICAL coordinates (U = C_n^-T W_hat) so the NEXT step
    % -- whose ILDL factor is rebuilt from scratch -- recycles the SAME physical
    % subspace rather than the same numbers in a different coordinate system.  The
    % columns are raw, unnormalized MINRES residuals spanning many orders of
    % magnitude; applyCtinv is linear, and augment_recycle_V removes the scale at
    % the next step before its rank test.  The key is renamed krylov_W -> krylov_U
    % so a stale cache from a pre-transport session cannot be read as if it were
    % already physical.
    pc.cache('krylov_U') = struct('step', pc.step, 'U', P.applyCtinv(getU()));
end

function [x, fl, rr, it] = tl_solve_lowrank(K, b, tol, mit, pc, opts, R_ildl, R_ref)
%TL_SOLVE_LOWRANK  Split two-level solve whose coarse space is a randomized sketch of
% the update operator D = K_ref^-1 (K_n - K_ref).  See build_lowrank_sketch_V for the
% method and the span it targets; everything downstream of V is the ordinary
% src.precond.two_level_split_solve the rest of the family uses.
%
% WHAT IS RECYCLED IS THE FACTORIZATION, NOT THE SUBSPACE.  Every other basis in this
% file is cached across steps and transported into the current coordinates
% (cached_basis).  Here V depends on dK = K_n - K_ref, which changes every step, so
% there is nothing to freeze: the frozen object is the reference ldl, and V is rebuilt
% from it each step.  Using cached_basis would silently pin the coarse space to the
% step it was first built at and measure something else entirely.
%
% The ILDL smoother comes from the SAME cache key the other two-level entries use, so
% adding this arm costs no extra smoother -- only its own coarse space.
%
% VALIDITY PREDICATE on the reference context: required for the same reason
% exact_ldl_frozen needs one.  A frozen factor is APPLIED at every later step and
% size(K,1) = nU + nP + nC changes if a Lagrange point leaves the fluid mesh; without
% it, dK = K - ctx.Kref is an opaque dimension error rather than a warned rebuild.
%
% info is stashed under 'lowrank_info' (a plain cache set, not the refresh-cadence
% path) so the EFFECTIVE coarse dimension -- info.ncols, which rank truncation can put
% below k -- is observable after a run.  k is what was asked for; ncols is what was
% deflated, and it is the one to quote.
    P   = cached_ildl(K, pc, opts, R_ildl);
    ctx = cached(pc, 'lowrank_ref', R_ref, ...
                 @() frozen_ldl_context(K), ...
                 @(v) v.n == size(K, 1));

    o = struct('k',      opts.oversample * opts.sm_eig, ...
               'q',      opts.q, ...
               'reorth', opts.reorth, ...
               'Cn',     current_C(pc, P));
    [V, info] = build_lowrank_sketch_V(ctx, K, P, o);
    pc.cache('lowrank_info') = struct('step', pc.step, 'val', info);

    [x, fl, rr, it] = src.precond.two_level_split_solve(K, b, tol, mit, P, V, opts.tau);
end

function P = cached_ildl(K, pc, opts, R_ildl)
%CACHED_ILDL  The ILDL smoother on its own refresh cadence, under the key every
% two-level entry shares (so it is built at most once per refresh-step no matter how
% many solvers ask for it).
    ikey = ['ildl_' opts.ildl_mode];
    ildl_opts = struct('mode', opts.ildl_mode);
    if strcmp(opts.ildl_mode, 'droptol'), ildl_opts.droptol = opts.droptol; end
    P = cached(pc, ikey, R_ildl, ...
               @() src.precond.make_ildl_precond(K, ildl_opts));
end

function [P, V] = two_level_parts(K, pc, method, opts, R_ildl, R_deflat, R_dinv)
%TWO_LEVEL_PARTS  The cached ILDL factor and coarse basis for one V-building method.
% Shared by every two-level entry, so the ildl / dinv / V_<method> keys are built at
% most once per refresh-step no matter how many solvers ask for them.
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
% two_level_parts is called once per two-level solver entry per step (sjlt,
% gaussian, polynomial, exact) AND a second time for 'gaussian' via
% tl_solve_krylov, so without .hstep the C^T multiply + QR would run up to five
% times per step.  Cost of the memo is one extra n-by-k block per method.
%
% At a (re)build step nothing is transported: build_deflation_V already returns an
% orthonormal V in the CURRENT step's hat coordinates, so it is returned verbatim
% and we pay only one applyCtinv to stash U.  Round-tripping through C^-T then C^T
% would be both wasteful and slightly less accurate.
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
        warning('define_solver_list:basisRankDrop', ...
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
% promoted to +src.  Delete this call (and the transport wiring) to revert.
    if ~isempty(which('transport_V')), return; end
    thisDir   = fileparts(mfilename('fullpath'));
    kernelDir = fullfile(fileparts(thisDir), 'linear_solves', ...
                         'subspace_recycle', 'kernel');
    if exist(kernelDir, 'dir')
        addpath(kernelDir);
    else
        error('define_solver_list:noKernel', ...
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
% mesh, the same hazard cached_basis guards with size(e.U,1) ~= n.  Omit it and the
% cadence logic is character-for-character what it was, which is how the ildl /
% dinv / blockjac_Lc call sites keep their behaviour exactly.
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
        warning('define_solver_list:cacheShapeChanged', ...
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
function y = block_precond(r, nU, nP, nC, Lc, Rp, nu)
%BLOCK_PRECOND  Apply the SPD block-diagonal ("block Jacobi") preconditioner
% P^{-1} r for the Stokes KKT system.
%   Pu   ~ Avel                          (applied via ichol factor Lc)
%   Pp   ~ (1/nu) * pressure mass        (applied via chol factor Rp) -> nu*M^{-1}
%   Plam = I
    ru = r(1:nU);
    rp = r(nU + (1:nP));
    rl = r(nU + nP + (1:nC));

    yu = Lc' \ (Lc \ ru);
    yp = nu * (Rp \ (Rp' \ rp));
    yl = rl;

    y = [yu; yp; yl];
end
