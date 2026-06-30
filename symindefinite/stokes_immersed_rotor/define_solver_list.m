function solvers = define_solver_list(params)
%DEFINE_SOLVER_LIST  MINRES solver/preconditioner registry for the
% Stokes-immersed-rotor benchmark (simplified deal.II step-70).
%
%   solvers = define_solver_list(params)
%
%   Returns a cell array of solver structs.  Each per-step KKT system is
%   SYMMETRIC INDEFINITE and is solved with MINRES; a solver entry differs only
%   in the (SPD) preconditioner it applies, or supplies its own self-contained
%   solve.  This is the extensibility seam: adding a preconditioner is a single
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
%             on the split operator C^-1 K C^-T and unwinds.
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
%
%   Two-level / deflation method knobs (params.*; defaults mirror
%   report/ball_surface/run_benchmark.m):
%     DEFLAT_SM_EIG        # smallest-|lambda| deflation vectors (report sm_eig)
%     DEFLAT_LG_EIG        # largest-|lambda|  deflation vectors (report lg_eig)
%     DEFLAT_Q             sketch power-iteration rounds (gaussian/sjlt V)
%     DEFLAT_TAU           deflation coarse-correction weight tau
%     DEFLAT_CHEB_DEGREE   Chebyshev degree (polynomial V; exact eigs band)
%     ILDL_MODE            incomplete-LDL pattern: 'nofill' | 'droptol'
%     ILDL_DROPTOL         drop tolerance when ILDL_MODE = 'droptol'
%
%   Mirrors define_motion_list (the geometry/motion registry).  Per the project
%   convention, this preconditioner registry stays LOCAL to the benchmark (it
%   changes/grows and is not geometry-persistent); the +src engine is kept
%   preconditioner-agnostic.

    if nargin < 1 || isempty(params), params = struct(); end

    % --- per-preconditioner refresh cadences (report-style; Inf = build once) ---
    R_blkjac = getdef(params, 'BLOCKJAC_PREC_REFRESH', Inf);
    R_ildl   = getdef(params, 'ILDL_PREC_REFRESH',     1);
    R_deflat = getdef(params, 'DEFLAT_PREC_REFRESH',   Inf);
    R_dinv   = getdef(params, 'DINVERSE_PREC_REFRESH', Inf);

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
        'cheb_degree',  getdef(params, 'DEFLAT_CHEB_DEGREE',  12));

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

    % Two-level deflation (ILDL smoother + indefinite deflation, B = L^-T P L^-1),
    % one entry per V-building operation.  'exact' last -> featured in accuracy.png.
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
    ikey = ['ildl_' opts.ildl_mode];
    ildl_opts = struct('mode', opts.ildl_mode);
    if strcmp(opts.ildl_mode, 'droptol'), ildl_opts.droptol = opts.droptol; end
    P = cached(pc, ikey, R_ildl, ...
               @() src.precond.make_ildl_precond(K, ildl_opts));

    if strcmp(method, 'none')
        V = [];
    else
        dA = [];
        if any(strcmp(method, {'gaussian', 'sjlt'}))   % only inverse-power needs a factor
            dA = cached(pc, 'dinv', R_dinv, @() decomposition(K));
        end
        o = opts;  o.method = method;
        V = cached(pc, ['V_' method], R_deflat, ...
                   @() src.precond.build_deflation_V(K, P, o, dA));
    end

    [x, fl, rr, it] = src.precond.two_level_split_solve(K, b, tol, mit, P, V, opts.tau);
end

function v = cached(pc, key, refresh, buildFn)
%CACHED  Per-component refresh cache keyed in pc.cache (one rebuild at most per
% step, on the mod(step-1,refresh)==0 cadence).  Shared keys (ildl/dinv) are thus
% built once per refresh-step and reused across solver entries within that step.
    c = pc.cache;
    if ~isKey(c, key)
        v = buildFn();
        c(key) = struct('step', pc.step, 'val', v);
        return;
    end
    e = c(key);
    if e.step ~= pc.step && mod(pc.step - 1, refresh) == 0
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
