%RUN_AMG_SUBSPACE_CAPTURE_TSYM  AMG subspace-capture ablation on Tsym.
%
% Same ablation as run_amg_subspace_capture.m, but the TARGET SYSTEM is the
% ichol split-preconditioned operator (as in subspace_capture/):
%     Tsym = L^{-1} A L^{-T},   L = ichol(A,'nofill'),
% i.e. the system the two-level deflated split scheme actually runs on.
% Question: do the A-targeted ablation conclusions (SA vs SJLT projector gap,
% smoothing / inner-solve sensitivity, coarse-size sweep) survive when the
% object of study is the preconditioned system instead of A itself?
%
% Operator construction (the only substantive change vs the A-targeted run):
%   * The AMG hierarchy is built on A exactly as before (make_amg_prec_ablate
%     with the ichol fine smoother); M ~= A^{-1}.
%   * Its conjugation  Zhat = L^T M L  ~=  L^T A^{-1} L = Tsym^{-1}  is the
%     matching approximate inverse of Tsym, so the subspace iteration becomes
%         V0 = randn(n, m);   V <- orth(Lt * M(L * V))   (q times).
%     Zhat inherits M's symmetry caveat: (L^T M L)' = L^T M' L, so Zhat is
%     symmetric iff preSmooth == postSmooth.
%   * Reference config 'exact_inverse' uses the exact Tsym^{-1} = L^T A^{-1} L.
%   * Ground truth = smallest k eigenvectors of Tsym (NOT of A); kappa is the
%     two-level deflated condition number of Tsym with tau = lam_{k+1}(Tsym).
%   * The wrap adds two sparse matvecs with L per apply; the work-unit proxy
%     is adjusted by 2*nnz(L)/nnz(A) accordingly (still 1 WU = one fine
%     matvec with A).
%
% Configs, metrics, CSV schemas, plot folders and file names are IDENTICAL to
% run_amg_subspace_capture.m so output_tsym/ is diffable side-by-side against
% output/.  See that script's header for the ablation rationale (sections
% A_smoothing / B_inner_solve / C_projector / D_coarse_size, references
% sa_pre1_post1_chol + exact_inverse + exact_inverse_plain drawn in every
% section; the _plain twin iterates the exact inverse WITHOUT per-step
% re-orthonormalization and demonstrates the roundoff rank collapse that
% truncates subspace_capture/run_inverse_subspace_iter.m past q ~ 4).
%
% Outputs (amg_subspace/output_tsym/ or output_tsym_smoke/):
%   results.mat / results.csv        one row per (config, q), all sections
%   setup_apply_cost.csv             one row per config
%   <section>/                       eigspace_err2_vs_q.pdf,
%                                    angle_capture_fraction_vs_q.pdf,
%                                    kappa_ratio_vs_q.pdf, aggregate_1x3.png
%
% Usage:
%   cd amg_subspace
%   run_amg_subspace_capture_tsym

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);                                 % src.* packages
addpath(fullfile(repoRoot, 'subspace_capture'));   % capture + kappa metrics
addpath(thisFileDir);                              % local AMG factory

%% --- Configuration ---------------------------------------------------------
% true = fast end-to-end check (small mesh, 4 configs).  Pre-set `smoke` in
% the workspace (e.g. `smoke = true; run_amg_subspace_capture_tsym`) to
% override.
if ~exist('smoke', 'var'), smoke = false; end

if smoke
    h0        = 0.15;
    k         = 40;
    iters     = [0 1 2 4];
    minCoarse = 100;
    outDir    = fullfile(thisFileDir, 'output_tsym_smoke');
else
    h0        = 0.05;
    k         = 200;
    iters     = [0 1 2 3 4 5 6 8 10 12 16 20];
    minCoarse = 800;
    outDir    = fullfile(thisFileDir, 'output_tsym');
end
m        = 2 * k;
cacheDir = fullfile(outDir, 'cache');
if ~isfolder(outDir),   mkdir(outDir);   end
if ~isfolder(cacheDir), mkdir(cacheDir); end

% Snapshot (identical to subspace_capture/run_inverse_subspace_iter.m).
contrast    = 60;
t_snap      = 0;
dt          = 1;
Tmax        = 100;
mesh_method = 'pdetoolbox';

% Fixed AMG structure.
theta       = 0.05;
maxAggSize  = 16;
omegaSmooth = 2/3;
seed        = 1;

configs = make_configs(smoke);

%% --- Build snapshot A + ichol factor --------------------------------------
fprintf('\n--- Building sphere snapshot (h0=%.4g, %s) ---\n', h0, mesh_method);
[A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
n  = msh.numIN;
Lt = L';
fprintf('A: %d x %d, nnz=%d, sym=%d   nnz(L)=%d\n', ...
        size(A,1), size(A,2), nnz(A), issymmetric(A), nnz(L));

%% --- Target operator Tsym = L^{-1} A L^{-T} --------------------------------
dA      = decomposition(A, 'chol');
Tfun    = @(X) L \ (A * (Lt \ X));        % forward Tsym
Tinvfun = @(X) Lt * (dA \ (L * X));       % exact Tsym^{-1} = L^T A^{-1} L

%% --- Ground truth: smallest k eigenvectors of Tsym --------------------------
[V_true, lam_cut] = load_or_compute_eigs_Tsym(cacheDir, Tinvfun, n, k, h0);
fprintf('V_true (smallest %d eigvecs of Tsym): %d x %d  (lam_cut=%.4e)\n', ...
        k, size(V_true,1), size(V_true,2), lam_cut);

%% --- kappa_exact: best-achievable deflated condition number ----------------
condOpts    = struct('eigs_tol', 1e-8, 'eigs_maxit', 5000, 'W_is_orth', true);
% Raw-block variant (no W_is_orth): the un-reorthogonalized 'plain' config
% passes a non-orthonormal V, whose internal pivoted QR records the rank.
condOptsRaw = struct('eigs_tol', 1e-8, 'eigs_maxit', 5000);
exactCond = deflated_cond_two_level(V_true, Tfun, Tinvfun, lam_cut, n, condOpts);
if ~exactCond.ok
    error('run_amg_subspace_capture_tsym:kappaExactFailed', ...
          'kappa_exact computation failed: %s', exactCond.err);
end
kappa_exact = exactCond.kappa;
lam_max_T   = exactCond.lam_max;
fprintf('kappa_exact = %.6e  (analytic lam_max_T/lam_cut = %.6e)\n', ...
        kappa_exact, lam_max_T / lam_cut);

%% --- Shared Gaussian starting block ---------------------------------------
rng(seed);
V0 = randn(n, m);

%% --- Sweep over AMG configurations ----------------------------------------
ops = struct('Afun', Tfun, 'Ainvfun', Tinvfun, 'lam_cut', lam_cut, ...
             'n', n, 'm', m, 'kappa_exact', kappa_exact, ...
             'condOpts', condOpts, 'condOptsRaw', condOptsRaw, ...
             'iters', iters);

% Resolve the matched coarse size for the '_ncmatch' sjlt configs: the SA
% baseline's realized nc, read off a cheap throwaway hierarchy build.
[~, saInfo] = make_amg_prec_ablate(A, ...
    'maxLevels', 2, 'minCoarseSize', minCoarse, ...
    'theta', theta, 'maxAggSize', maxAggSize, ...
    'omegaInterp', 0, 'omegaSmooth', omegaSmooth, ...
    'fineSmootherL', L, 'fineSmootherLt', Lt);
nc_sa = saInfo.coarseN;
fprintf('SA realized coarse size nc_sa = %d (used by the _ncmatch configs)\n', ...
        nc_sa);
for ic = 1:numel(configs)
    if strcmp(configs(ic).projector, 'sjlt') && isnan(configs(ic).sjltNc)
        configs(ic).sjltNc = nc_sa;
        configs(ic).label  = sprintf('%s (nc=%d)', configs(ic).name, nc_sa);
    end
end

% Extra WU per Zhat apply: the two sparse matvecs with L in Lt * M(L * X).
wrapWU = 2 * nnz(L) / nnz(A);

rows = [];
for ic = 1:numel(configs)
    cfg = configs(ic);
    fprintf('\n[%d/%d] %s  (proj=%s pre=%d post=%d coarse=%s)\n', ...
            ic, numel(configs), cfg.name, cfg.projector, ...
            cfg.preSmooth, cfg.postSmooth, cfg.coarseSolve);

    if strcmp(cfg.projector, 'exact')
        % Ideal-operator reference: exact Tsym^{-1} = L^T A^{-1} L via the
        % chol decomposition.  The AMG cost model does not apply (NaN work
        % proxies); measured apply time is still recorded by the sweep.
        Mfun  = Tinvfun;
        ainfo = struct('projector', 'exact', 'sjltNnzPerCol', NaN, ...
                       'levels', struct('n', n, 'nnzA', nnz(A), 'nnzP', 0), ...
                       'nLevels', 1, 'coarseN', NaN, 'coarseNnz', NaN, ...
                       'coarseType', 'exact', 'setupTime', NaN, ...
                       'workPerApply', NaN, 'workUnits', NaN);
        fprintf('  exact inverse Tsym^{-1} = L''A^{-1}L (chol reference)\n');
    else
        % Reseed per config: sjlt draws become reproducible AND distinct.
        rng(seed + ic);
        [Mamg, ainfo] = make_amg_prec_ablate(A, ...
            'maxLevels', 2, 'minCoarseSize', minCoarse, ...
            'theta', theta, 'maxAggSize', maxAggSize, ...
            'omegaInterp', 0, 'omegaSmooth', omegaSmooth, ...
            'preSmooth', cfg.preSmooth, 'postSmooth', cfg.postSmooth, ...
            'coarseSolve', cfg.coarseSolve, ...
            'coarseJacobiSweeps', cfg.coarseJacobiSweeps, ...
            'coarsePcgTol', cfg.coarsePcgTol, ...
            'coarsePcgMaxit', cfg.coarsePcgMaxit, ...
            'projector', cfg.projector, ...
            'sjltNc', cfg.sjltNc, 'sjltNnzPerCol', cfg.sjltNnzPerCol, ...
            'fineSmootherL', L, 'fineSmootherLt', Lt);

        % Conjugated V-cycle: Zhat = L^T M L ~= Tsym^{-1}.
        Mfun = @(X) Lt * (Mamg(L * X));
        ainfo.workUnits = ainfo.workUnits + wrapWU;

        fprintf('  levels %s  coarse=%s  setup=%.2fs  WU/apply=%.2f\n', ...
                mat2str([ainfo.levels.n]), ainfo.coarseType, ...
                ainfo.setupTime, ainfo.workUnits);
    end
    if cfg.preSmooth == 0 && cfg.postSmooth == 0 && ainfo.coarseN < m
        warning('run_amg_subspace_capture_tsym:rankLimited', ...
                ['config %s: pure coarse-grid correction with coarseN=%d < ', ...
                 'm=%d -- captured subspace is rank-limited.'], ...
                cfg.name, ainfo.coarseN, m);
    end

    cfgRows = run_one_config(cfg, Mfun, ainfo, V0, V_true, ops);
    rows = [rows; cfgRows];                                       %#ok<AGROW>
end

%% --- Save ------------------------------------------------------------------
meta = struct('target', 'Tsym', 'n', n, 'k', k, 'm', m, 'iters', iters, ...
              'h0', h0, 'contrast', contrast, 't_snap', t_snap, 'dt', dt, ...
              'seed', seed, 'theta', theta, 'maxAggSize', maxAggSize, ...
              'omegaSmooth', omegaSmooth, 'minCoarseSize', minCoarse, ...
              'maxLevels', 2, 'lam_cut', lam_cut, 'lam_max_Tsym', lam_max_T, ...
              'kappa_exact', kappa_exact, 'smoke', smoke, ...
              'nc_sa', nc_sa, 'config_names', {{configs.name}});
save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');
write_results_csv(fullfile(outDir, 'results.csv'), rows);
write_cost_csv(fullfile(outDir, 'setup_apply_cost.csv'), rows, iters);
fprintf('\nresults.mat / results.csv / setup_apply_cost.csv written to:\n  %s\n', ...
        outDir);

%% --- Plots -----------------------------------------------------------------
fprintf('\n--- Rendering plots ---\n');
make_amg_capture_plots(rows, configs, outDir);
fprintf('\nDone.\n');

%% =========================================================================
%% Local helpers
%% =========================================================================
function configs = make_configs(smoke)
%MAKE_CONFIGS  The ablation grid: 19 configs, all maxLevels = 2.
%   Identical to run_amg_subspace_capture.m (kept in sync by hand so the
%   Tsym-targeted results are config-for-config comparable).
    C = struct('name', {}, 'preSmooth', {}, 'postSmooth', {}, ...
               'coarseSolve', {}, 'coarseJacobiSweeps', {}, ...
               'coarsePcgTol', {}, 'coarsePcgMaxit', {}, ...
               'projector', {}, 'sjltNc', {}, 'sjltNnzPerCol', {}, ...
               'section', {}, 'label', {}, 'color', {}, 'style', {});

    add = @(C, name, pre, post, cs, nu, tol, mit, proj, nc, s, sect, col, st) ...
        [C, struct('name', name, 'preSmooth', pre, 'postSmooth', post, ...
                   'coarseSolve', cs, 'coarseJacobiSweeps', nu, ...
                   'coarsePcgTol', tol, 'coarsePcgMaxit', mit, ...
                   'projector', proj, 'sjltNc', nc, 'sjltNnzPerCol', s, ...
                   'section', sect, 'label', name, 'color', col, 'style', st)];

    % --- references (drawn in every section) -------------------------------
    % exact_inverse here is the exact Tsym^{-1} = L^T A^{-1} L, the
    % ideal-operator floor V <- orth(Tsym^{-1} V).
    C = add(C, 'sa_pre1_post1_chol', 1, 1, 'chol', NaN, NaN, NaN, ...
            'sa', NaN, NaN, 'baseline', [0.00 0.00 0.00], '-');
    C = add(C, 'exact_inverse', NaN, NaN, 'exact', NaN, NaN, NaN, ...
            'exact', NaN, NaN, 'baseline', [0.45 0.45 0.45], ':');
    % --- A_smoothing: (pre,post) sweep, SA, exact coarse -------------------
    C = add(C, 'sa_pre0_post0_chol', 0, 0, 'chol', NaN, NaN, NaN, ...
            'sa', NaN, NaN, 'A_smoothing', [0.55 0.55 0.55], '-');
    C = add(C, 'sa_pre0_post1_chol', 0, 1, 'chol', NaN, NaN, NaN, ...
            'sa', NaN, NaN, 'A_smoothing', [0.00 0.45 0.74], '-');
    C = add(C, 'sa_pre1_post0_chol', 1, 0, 'chol', NaN, NaN, NaN, ...
            'sa', NaN, NaN, 'A_smoothing', [0.30 0.75 0.93], '-');
    C = add(C, 'sa_pre2_post2_chol', 2, 2, 'chol', NaN, NaN, NaN, ...
            'sa', NaN, NaN, 'A_smoothing', [0.47 0.67 0.19], '-');
    C = add(C, 'sa_pre3_post3_chol', 3, 3, 'chol', NaN, NaN, NaN, ...
            'sa', NaN, NaN, 'A_smoothing', [0.13 0.40 0.13], '-');
    % --- B_inner_solve: coarse solve at pre=post=1 (+ cross configs) -------
    C = add(C, 'sa_pre1_post1_jac2',    1, 1, 'jacobi', 2,  NaN,  NaN, ...
            'sa', NaN, NaN, 'B_inner_solve', [0.85 0.33 0.10], '-');
    C = add(C, 'sa_pre1_post1_jac10',   1, 1, 'jacobi', 10, NaN,  NaN, ...
            'sa', NaN, NaN, 'B_inner_solve', [0.64 0.08 0.18], '-');
    C = add(C, 'sa_pre1_post1_pcg1e-2', 1, 1, 'pcg',    NaN, 1e-2, 50, ...
            'sa', NaN, NaN, 'B_inner_solve', [0.93 0.69 0.13], '-');
    C = add(C, 'sa_pre0_post1_jac2', 0, 1, 'jacobi', 2, NaN, NaN, ...
            'sa', NaN, NaN, 'B_inner_solve', [0.49 0.18 0.56], '-.');
    C = add(C, 'sa_pre2_post2_jac2', 2, 2, 'jacobi', 2, NaN, NaN, ...
            'sa', NaN, NaN, 'B_inner_solve', [0.75 0.45 0.75], '-.');
    % --- C_projector: SA vs SJLT at matched nc (sjltNc = NaN -> resolved to
    %     the SA baseline's realized nc before the sweep) -------------------
    C = add(C, 'sjlt4_pre1_post1_chol_ncmatch', 1, 1, 'chol', NaN, NaN, NaN, ...
            'sjlt', NaN, 4, 'C_projector', [0.85 0.33 0.10], '--');
    C = add(C, 'sjlt1_pre1_post1_chol_ncmatch', 1, 1, 'chol', NaN, NaN, NaN, ...
            'sjlt', NaN, 1, 'C_projector', [0.93 0.69 0.13], '--');
    % --- D_coarse_size: SJLT (s=4) nc sweep, warm -> cool with growing nc --
    ncSweep   = [100 200 400 800 1600 3200];
    ncColors  = [0.99 0.75 0.15;  0.90 0.50 0.15;  0.80 0.30 0.30; ...
                 0.55 0.25 0.55;  0.30 0.35 0.75;  0.10 0.55 0.85];
    for iv = 1:numel(ncSweep)
        C = add(C, sprintf('sjlt4_pre1_post1_chol_nc%04d', ncSweep(iv)), ...
                1, 1, 'chol', NaN, NaN, NaN, ...
                'sjlt', ncSweep(iv), 4, 'D_coarse_size', ncColors(iv,:), '--');
    end

    % Every config above re-orthonormalizes after each apply.  The _plain
    % twin of the exact-inverse reference iterates WITHOUT per-step orth
    % (like src.precond.subspace_iter_plain): the block numerically loses
    % rank as all columns align with the dominant directions -- the failure
    % mode that truncates subspace_capture's inverse-iteration curves.
    [C.reorth]      = deal(true);
    plainCfg        = C(strcmp({C.name}, 'exact_inverse'));
    plainCfg.name   = 'exact_inverse_plain';
    plainCfg.label  = 'exact_inverse_plain';
    plainCfg.color  = [0.70 0.70 0.70];
    plainCfg.reorth = false;
    C = [C, plainCfg];

    if smoke
        keep = ismember({C.name}, ...
                        {'sa_pre1_post1_chol', 'exact_inverse', ...
                         'exact_inverse_plain', ...
                         'sa_pre0_post1_chol', 'sa_pre1_post1_jac2', ...
                         'sjlt4_pre1_post1_chol_ncmatch'});
        C = C(keep);
    end
    configs = C;
end

function rows = run_one_config(cfg, Mfun, ainfo, V0, V_true, ops)
%RUN_ONE_CONFIG  One incremental q-sweep: V <- orth(Mfun(V)), metrics at each
%   recorded q.  cfg.reorth = true re-orthonormalizes EVERY step: Zhat's
%   spectrum decays like 1/lambda(Tsym), so an unorthogonalized block loses
%   the trailing directions to roundoff well before q = 20.  cfg.reorth =
%   false skips the in-loop orth (plain iteration); the metrics then
%   orthonormalize a rank-truncating copy, so r_comp/r_defl record the
%   collapse.
    iters   = ops.iters;
    V       = orth(V0);
    q_prev  = 0;
    t_apply = 0;
    t_orth  = 0;
    rows    = [];

    for iq = 1:numel(iters)
        q = iters(iq);
        for s = q_prev+1 : q
            t0 = tic;  V = Mfun(V);  t_apply = t_apply + toc(t0);
            if cfg.reorth
                t1 = tic;  V = orth(V);  t_orth = t_orth + toc(t1);
            end
        end
        q_prev = q;

        row = new_row(cfg, ainfo, q);
        row.apply_time_cum = t_apply;
        row.orth_time_cum  = t_orth;
        row.work_units_cum = q * ops.m * ainfo.workUnits;
        try
            cap = subspace_capture_directed(V_true, V, [], ...
                      struct('true_is_orth', true, ...
                             'comp_is_orth', cfg.reorth));
            row.eigspace_err_2          = cap.eigspace_err_2;
            row.eigspace_err_fro        = cap.eigspace_err_fro;
            row.n_angle_below_1pct      = cap.n_angle_below_1pct;
            row.n_angle_below_0p1pct    = cap.n_angle_below_0p1pct;
            row.angle_capture_frac_1pct = cap.n_angle_below_1pct ...
                                          / max(cap.r_true, 1);
            row.r_true = cap.r_true;
            row.r_comp = cap.r_comp;
            row.ok     = true;
        catch ME
            row.err = regexprep(ME.message, '\n.*', '');
            warning('run_amg_subspace_capture_tsym:captureFailed', ...
                    '%s q=%d capture failed: %s', cfg.name, q, row.err);
        end

        % Deflated condition number never throws (ok=false on failure).
        if cfg.reorth, co = ops.condOpts; else, co = ops.condOptsRaw; end
        c = deflated_cond_two_level(V, ops.Afun, ops.Ainvfun, ...
                                    ops.lam_cut, ops.n, co);
        row.kappa       = c.kappa;
        row.kappa_ratio = c.kappa / ops.kappa_exact;
        row.r_defl      = c.r;
        row.kappa_ok    = c.ok;

        rows = [rows; row];                                       %#ok<AGROW>
        fprintf(['  q=%2d : err_2=%.3e  frac_1pct=%.3f  kappa_ratio=%.3e', ...
                 '  t_apply=%.1fs\n'], ...
                q, row.eigspace_err_2, row.angle_capture_frac_1pct, ...
                row.kappa_ratio, row.apply_time_cum);
    end
end

function row = new_row(cfg, ainfo, q)
%NEW_ROW  Result row pre-filled with config/cost identifiers and NaN metrics.
    row = struct( ...
        'config', cfg.name, 'section', cfg.section, ...
        'projector', cfg.projector, ...
        'sjltNc', cfg.sjltNc, 'sjltNnzPerCol', cfg.sjltNnzPerCol, ...
        'preSmooth', cfg.preSmooth, 'postSmooth', cfg.postSmooth, ...
        'coarseSolve', cfg.coarseSolve, ...
        'coarseJacobiSweeps', cfg.coarseJacobiSweeps, ...
        'reorth', cfg.reorth, ...
        'nLevels', ainfo.nLevels, 'coarseN', ainfo.coarseN, ...
        'coarseNnz', ainfo.coarseNnz, ...
        'q', q, ...
        'eigspace_err_2', NaN, 'eigspace_err_fro', NaN, ...
        'angle_capture_frac_1pct', NaN, ...
        'n_angle_below_1pct', NaN, 'n_angle_below_0p1pct', NaN, ...
        'r_true', NaN, 'r_comp', NaN, ...
        'kappa', NaN, 'kappa_ratio', NaN, 'r_defl', NaN, 'kappa_ok', false, ...
        'setup_time', ainfo.setupTime, ...
        'apply_time_cum', NaN, 'orth_time_cum', NaN, ...
        'work_units_cum', NaN, ...
        'work_units_per_apply', ainfo.workUnits, ...
        'ok', false, 'err', '');
end

function [A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method)
%BUILD_SNAPSHOT  Mesh + A = D_II + dt*K_II + L = ichol(A,'nofill').
%   Copied from subspace_capture/run_inverse_subspace_iter.m.
    msh      = src.discretization.build_sphere_mesh(h0, false, mesh_method);
    kappaFun = make_latitude_banding_contrast(Tmax, contrast);
    A        = assemble_snapshot_A(msh, kappaFun, dt, t_snap);
    L        = ichol_with_fallback(A);
end

function A = assemble_snapshot_A(msh, kappaFun, dt, tcur)
%ASSEMBLE_SNAPSHOT_A  Build A = D_II + dt*K_II at one time level (closed sphere).
    numIN   = msh.numIN;
    kappa_e = kappaFun(msh.cent(:,1), msh.cent(:,2), msh.cent(:,3), tcur);
    Vscale  = msh.Vunit .* repelem(kappa_e, 9);
    Vii     = Vscale(msh.idxII);
    K_II    = sparse(msh.I_II, msh.J_II, Vii, numIN, numIN);
    A       = msh.D_II + dt * K_II;
    A       = 0.5 * (A + A.');
end

function L = ichol_with_fallback(A)
%ICHOL_WITH_FALLBACK  ichol(A,'nofill') with diagcomp safety net.
    try
        L = ichol(A, struct('type', 'nofill'));
    catch
        alpha    = max(sum(abs(A), 2) ./ diag(A)) - 2;
        diagcomp = max(alpha, 0);
        L = ichol(A, struct('type', 'nofill', 'diagcomp', diagcomp));
    end
end

function [V_true, lam_cut] = load_or_compute_eigs_Tsym(cacheDir, Tinvfun, n, k, h0)
%LOAD_OR_COMPUTE_EIGS_TSYM  Smallest k+1 eigenpairs of Tsym (cached).
%   Cache key includes h0 and k so smoke and full runs never collide.  The
%   subspace_capture/output/cache/eigsTsym_k*.mat files are NOT h0-keyed and
%   are not reused here.
    cachePath = fullfile(cacheDir, sprintf('eigsTsym_h%g_k%d.mat', h0, k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_true  = S.V(:, 1:k);
        lam_cut = S.D(k + 1);
        fprintf('Loaded cached %s\n', cachePath);
        return;
    end
    fprintf('Computing smallest %d eigenpairs of Tsym (one-time)...\n', k + 1);
    t0 = tic;
    % Name-value options, NOT the legacy opts struct: eigs silently ignores
    % struct fields with these capitalized names.  With a function handle and
    % 'smallestabs', eigs expects the handle to apply Tsym^{-1} x.
    [Vraw, Dmat] = eigs(Tinvfun, n, k + 1, 'smallestabs', ...
                        'Tolerance', 1e-10, 'MaxIterations', 5000, ...
                        'IsFunctionSymmetric', true);
    [D, idx] = sort(real(diag(Dmat)), 'ascend');
    V        = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_true  = V(:, 1:k);
    lam_cut = D(k + 1);
end

function write_results_csv(csvPath, rows)
%WRITE_RESULTS_CSV  Flat CSV of the sweep results.
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['config,section,projector,sjltNc,sjltNnzPerCol,', ...
                  'preSmooth,postSmooth,coarseSolve,', ...
                  'coarseJacobiSweeps,reorth,nLevels,coarseN,coarseNnz,q,', ...
                  'eigspace_err_2,eigspace_err_fro,angle_capture_frac_1pct,', ...
                  'n_angle_below_1pct,n_angle_below_0p1pct,r_true,r_comp,', ...
                  'kappa,kappa_ratio,r_defl,kappa_ok,', ...
                  'setup_time,apply_time_cum,orth_time_cum,', ...
                  'work_units_cum,work_units_per_apply,ok\n']);
    for i = 1:numel(rows)
        r = rows(i);
        fprintf(fid, ['%s,%s,%s,%g,%g,%d,%d,%s,%g,%d,%d,%d,%d,%d,', ...
                      '%g,%g,%g,%g,%g,%g,%g,', ...
                      '%g,%g,%g,%d,', ...
                      '%g,%g,%g,%g,%g,%d\n'], ...
                r.config, r.section, r.projector, r.sjltNc, ...
                r.sjltNnzPerCol, r.preSmooth, r.postSmooth, r.coarseSolve, ...
                r.coarseJacobiSweeps, r.reorth, ...
                r.nLevels, r.coarseN, r.coarseNnz, ...
                r.q, ...
                r.eigspace_err_2, r.eigspace_err_fro, ...
                r.angle_capture_frac_1pct, ...
                r.n_angle_below_1pct, r.n_angle_below_0p1pct, ...
                r.r_true, r.r_comp, ...
                r.kappa, r.kappa_ratio, r.r_defl, r.kappa_ok, ...
                r.setup_time, r.apply_time_cum, r.orth_time_cum, ...
                r.work_units_cum, r.work_units_per_apply, r.ok);
    end
    fclose(fid);
end

function write_cost_csv(csvPath, rows, iters)
%WRITE_COST_CSV  One cost row per config (from that config's last sweep row).
    qmax  = max(iters);
    names = unique({rows.config}, 'stable');
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['config,section,projector,sjltNc,sjltNnzPerCol,', ...
                  'preSmooth,postSmooth,coarseSolve,reorth,', ...
                  'nLevels,coarseN,coarseNnz,setup_time,', ...
                  'apply_time_per_block,work_units_per_apply\n']);
    for i = 1:numel(names)
        sub = rows(strcmp({rows.config}, names{i}));
        [~, ilast] = max([sub.q]);
        r = sub(ilast);
        fprintf(fid, '%s,%s,%s,%g,%g,%d,%d,%s,%d,%d,%d,%d,%g,%g,%g\n', ...
                r.config, r.section, r.projector, r.sjltNc, ...
                r.sjltNnzPerCol, r.preSmooth, r.postSmooth, r.coarseSolve, ...
                r.reorth, r.nLevels, r.coarseN, r.coarseNnz, r.setup_time, ...
                r.apply_time_cum / max(qmax, 1), r.work_units_per_apply);
    end
    fclose(fid);
end

%% =========================================================================
%% Rendering
%% =========================================================================
function make_amg_capture_plots(rows, configs, out_dir)
%MAKE_AMG_CAPTURE_PLOTS  One plot folder per ablation section.
%   Each section folder gets the SAME three metric panels + an aggregate PNG;
%   the baseline config (section 'baseline') is drawn in every section so the
%   four ablations are cross-comparable.  Sections with no member present in
%   rows (smoke subsets) are skipped.
    sections = {'A_smoothing', 'B_inner_solve', 'C_projector', 'D_coarse_size'};
    secTitles = { ...
        ['A: pre/post smoothing sweeps (SA projector, exact coarse solve)', ...
         '  --  target = Tsym'], ...
        ['B: coarse inner solve (SA projector, at/around pre=post=1)', ...
         '  --  target = Tsym'], ...
        ['C: projector -- SA aggregation vs SJLT sketch at matched nc', ...
         '  --  target = Tsym'], ...
        'D: SJLT (s=4) coarse-size sweep  --  target = Tsym'};

    ranConfigs = unique({rows.config});
    for is = 1:numel(sections)
        sec = sections{is};
        sel = (strcmp({configs.section}, sec) | ...
               strcmp({configs.section}, 'baseline')) ...
              & ismember({configs.name}, ranConfigs);
        subCfg = configs(sel);
        if ~any(strcmp({subCfg.section}, sec))
            continue;   % no member of this section was run (smoke subset)
        end
        secDir = fullfile(out_dir, sec);
        make_section_plots(rows, subCfg, secDir, secTitles{is});
    end
end

function make_section_plots(rows, configs, out_dir, sec_title)
%MAKE_SECTION_PLOTS  The three metric panels + aggregate for one section.
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    specs = struct( ...
        'metric', {'eigspace_err_2', 'angle_capture_frac_1pct', ...
                   'kappa_ratio'}, ...
        'xfield', {'q', 'q', 'q'}, ...
        'xlabel', {'number of subspace iterations q', ...
                   'number of subspace iterations q', ...
                   'number of subspace iterations q'}, ...
        'xscale', {'linear', 'linear', 'linear'}, ...
        'yscale', {'log', 'linear', 'log'}, ...
        'ylabel', {'||(I-P)Q_{true}||_2', ...
                   'captured directions (sin\theta < 1%)', ...
                   '\kappa_{approx} / \kappa_{exact}'}, ...
        'title',  {'eigenspace error (Tsym)', 'angle capture fraction (Tsym)', ...
                   'deflated condition-number ratio (Tsym)'}, ...
        'legloc', {'northeast', 'southeast', 'northeast'}, ...
        'tag',    {'eigspace_err2_vs_q', 'angle_capture_fraction_vs_q', ...
                   'kappa_ratio_vs_q'});

    for ip = 1:numel(specs)
        fig = figure('Visible', 'off', 'Units', 'inches', ...
                     'Position', [0 0 5.4 3.4], 'Color', 'w');
        draw_panel(axes(fig), rows, configs, specs(ip));
        outfile = fullfile(out_dir, [specs(ip).tag '.pdf']);
        exportgraphics(fig, outfile, 'ContentType', 'vector');
        close(fig);
        fprintf('Wrote %s\n', outfile);
    end

    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 16.5 3.8], 'Color', 'w');
    tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    for ip = 1:numel(specs)
        draw_panel(nexttile(tl), rows, configs, specs(ip));
    end
    title(tl, sec_title, 'FontWeight', 'bold', 'FontSize', 12);
    outfile = fullfile(out_dir, 'aggregate_1x3.png');
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
    fprintf('Wrote %s\n', outfile);
end

function draw_panel(ax, rows, configs, spec)
%DRAW_PANEL  One metric-vs-x panel with a series per config.
    hold(ax, 'on');
    handles = gobjects(1, numel(configs));
    keep    = false(1, numel(configs));
    yall    = [];

    for ic = 1:numel(configs)
        cfg  = configs(ic);
        sub  = rows(strcmp({rows.config}, cfg.name));
        if isempty(sub), continue; end
        [~, ord] = sort([sub.q]);
        sub = sub(ord);
        xs = [sub.(spec.xfield)];
        ys = [sub.(spec.metric)];
        handles(ic) = plot(ax, xs, ys, [cfg.style 'o'], ...
                           'Color', cfg.color, 'MarkerFaceColor', cfg.color, ...
                           'MarkerSize', 3.5, 'LineWidth', 1.2);
        keep(ic) = true;
        yall = [yall, ys];                                        %#ok<AGROW>
    end
    hold(ax, 'off');

    if strcmp(spec.metric, 'kappa_ratio')
        yline(ax, 1, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end

    set(ax, 'XScale', spec.xscale, 'YScale', spec.yscale, 'Box', 'on', ...
            'LineWidth', 0.6, 'FontSize', 9);
    if strcmp(spec.xfield, 'q')
        qs = unique([rows.q]);
        xlim(ax, [min(qs) - 0.5, max(qs) + 0.5]);
        set(ax, 'XTick', qs);
    end
    if strcmp(spec.yscale, 'linear') && contains(spec.metric, 'frac')
        ylim(ax, [-0.02, 1.05]);
        set(ax, 'YTick', 0:0.2:1);
    elseif strcmp(spec.yscale, 'log') && ~isempty(yall)
        yp = yall(isfinite(yall) & yall > 0);
        if ~isempty(yp)
            ylim(ax, [min(yp) * 0.5, max(yp) * 1.5]);
        end
    end

    xlabel(ax, spec.xlabel, 'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, spec.ylabel, 'FontSize', 10, 'FontWeight', 'bold');
    title(ax, spec.title, 'FontSize', 11, 'FontWeight', 'bold', ...
          'Interpreter', 'tex');

    lgd = legend(ax, handles(keep), {configs(keep).label}, ...
                 'Location', spec.legloc, 'Box', 'on', ...
                 'EdgeColor', [0.65 0.65 0.65], 'Color', 'white', ...
                 'FontSize', 6.5, 'NumColumns', 2, 'Interpreter', 'none');
    lgd.ItemTokenSize = [14, 6];
end

function f = make_latitude_banding_contrast(Tmax, contrast)
%MAKE_LATITUDE_BANDING_CONTRAST  Latitude-banding kappa with adjustable contrast.
%   Kept LOCAL (not in +src): kappa factories are experiment-specific.
    if nargin < 2 || isempty(contrast), contrast = 60; end
    if contrast < 1
        error('make_latitude_banding_contrast:badContrast', ...
              'contrast must be >= 1, got %g', contrast);
    end
    kmin = 1 / sqrt(contrast);
    kmax = sqrt(contrast);
    band_width = 0.25;
    freqs = [sqrt(2), sqrt(3), sqrt(5)];
    f = @(x, y, z, t) latitude_banding_eval(x, y, z, t, Tmax, kmin, kmax, ...
                                            band_width, freqs);
end

function val = latitude_banding_eval(x, y, z, t, Tmax, kmin, kmax, bw, freqs) %#ok<INUSL>
    theta = acos(max(min(z, 1), -1));   % colatitude [0, pi]
    bump = zeros(size(x));
    base_positions = [pi/4, pi/2, 3*pi/4];
    for k = 1:3
        center = base_positions(k) + 0.5 * sin(2 * pi * freqs(k) * t / Tmax);
        center = max(0.1, min(pi - 0.1, center));
        dist = abs(theta - center);
        bump = bump + 0.5 + 0.5 * tanh((bw - dist) / 0.1);
    end
    bump = bump / 3;
    bump = min(bump, 1);
    val = kmin + (kmax - kmin) * bump;
end
