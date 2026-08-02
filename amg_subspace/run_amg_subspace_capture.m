%RUN_AMG_SUBSPACE_CAPTURE  AMG-as-approximate-inverse subspace-capture ablation.
%
% Studies how well a 2-level smoothed-aggregation AMG V-cycle M ~= A^{-1}
% captures the SMALLEST eigenvector space of the sparse SPD FEM snapshot A
% when used as the operator of Gaussian-sketched subspace iteration:
%     V0 = randn(n, m);   V <- orth(M(V))   (q times).
% Motivation: build a deflation basis from preconditioner applies only
% (roughly matrix-free after the AMG setup).
%
% Fixed AMG structure (least-cost by design):
%   * maxLevels = 2 : fine level + ONE coarse level, nothing deeper.
%   * fine smoother = ichol(A,'nofill') (the factory applies it on level 1;
%     with 2 levels no Jacobi-smoothed intermediate levels exist).
%
% Four ablation sections (config naming:  <proj>_pre<i>_post<j>_<coarse>[_nc<size>],
% proj in {sa, sjlt1, sjlt4} where the digit is s = nnz per fine row):
%   A_smoothing   : (preSmooth, postSmooth) counts; SA projector, exact coarse.
%   B_inner_solve : coarse "inner" solve at pre=post=1: exact chol vs nu
%                   damped-Jacobi sweeps vs loose-tol PCG (+ two cross configs).
%   C_projector   : SA tentative prolongator vs SJLT sketch at MATCHED coarse
%                   size nc.  Both are sparse projections; SA has 1 nnz/row
%                   placed by connectivity, SJLT has s random +-1/sqrt(s)
%                   entries/row (s=1 is CountSketch = random aggregation).
%   D_coarse_size : SJLT (s=4) coarse-size sweep nc in {100..3200} -- the size
%                   of the coarse solve is SJLT's free parameter, unlike SA
%                   where nc is emergent (~n/5 here).
%
% Why sections C/D are interesting (two-level AMG theory):
%   * Two-level convergence needs the WEAK APPROXIMATION PROPERTY
%     (Ruge-Stuben; Vanek-Mandel-Brezina; Falgout-Vassilevski): range(P) must
%     approximate every algebraically smooth error with energy accuracy
%     ~ lambda/lambda_max, i.e. contain the near-kernel.  The SA tentative
%     prolongator is BUILT for this (piecewise-constant over connectivity
%     aggregates reproduces the Laplacian near-kernel).  Exact coarse-grid
%     correction is I - pi_A(range(P)): it removes only the error component
%     inside range(P), and a fixed smooth mode has expected captured energy
%     ~ nc/n in a RANDOM nc-dim subspace -- so theory predicts SJLT plateaus
%     far above SA at matched nc.  The SJLT embedding guarantee (nc ~ k
%     polylog) is about Omega'x preserving norms, NOT about range(Omega)
%     aligning with a chosen subspace; randomized eigensolvers get away with
%     nc ~ k only because they sketch THROUGH A^{-1} applies (which is what
%     the outer subspace iteration does), an alignment the coarse space
%     inside M never receives.
%   * Limits anchoring section D: nc -> n gives CGC -> Omega(Omega'A
%     Omega)^{-1}Omega' = A^{-1} exactly (unit-tested), i.e. exact inverse
%     iteration; nc -> 0 leaves only the ichol smoother, whose dominant
%     invariant subspace is NOT the small-lambda space.  The sweep reads off
%     the minimal nc with acceptable capture/kappa (e.g. kappa_ratio <~ 2 at
%     q = 20).
%   * Cost counterpoint: SJLT is trivially cheap to CONSTRUCT (no graph
%     traversal / strength-of-connection), but Omega'*A*Omega loses locality
%     and fills in, so its coarse chol is denser than SA's at the same nc --
%     setup_time / coarseNnz / work units in the CSVs test the "easier to
%     construct" claim end-to-end.
%
% Caveats (also encoded in make_amg_prec_ablate):
%   * preSmooth ~= postSmooth makes M NONSYMMETRIC.  Subspace iteration still
%     converges to the dominant invariant subspace of M, and the directed
%     principal-angle metric is basis-invariant, so the comparison stands --
%     but such an M is not a valid pcg preconditioner.
%   * preSmooth = postSmooth = 0 is pure coarse-grid correction with
%     rank(M) = coarseN; the captured space then lives inside an
%     coarseN-dimensional range (warned about if coarseN < m).
%
% The capture-error-vs-q curves plateau at the angle between M's dominant
% invariant subspace and the true smallest-k eigenspace of A; that plateau,
% per AMG configuration, is the quantity of interest.
%
% Metrics per (config, q):
%   * directed principal angles (subspace_capture_directed): eigspace_err_2,
%     eigspace_err_fro, capture fractions at sin(theta) < 1% / 0.1%;
%   * two-level deflated condition number kappa of A deflated with the
%     captured basis (deflated_cond_two_level with Tfun = A, tau =
%     lam_{k+1}(A)), reported as kappa / kappa_exact where kappa_exact uses
%     the true eigenvectors (analytically lam_max(A)/lam_{k+1});
%   * cost: AMG setup time, cumulative measured block-apply time, and a
%     flop-proxy work-unit count (1 WU = one fine matvec; see
%     make_amg_prec_ablate).  orth() time is tracked separately -- it is
%     identical across configs and not an AMG property.
%
% Outputs (amg_subspace/output/ or output_smoke/):
%   results.mat / results.csv        one row per (config, q), all sections
%   setup_apply_cost.csv             one row per config
%   <section>/                       one folder per ablation section
%     eigspace_err2_vs_q.pdf           ||(I-P)Q_true||_2 (log y) vs q
%     angle_capture_fraction_vs_q.pdf  fraction with sin(theta) < 1% vs q
%     kappa_ratio_vs_q.pdf             kappa/kappa_exact (log y) vs q
%     aggregate_1x3.png                the three panels side by side
% with section in {A_smoothing, B_inner_solve, C_projector, D_coarse_size};
% two reference series are drawn in EVERY section so the four ablations are
% cross-comparable:
%   sa_pre1_post1_chol (black solid)  -- the AMG baseline;
%   exact_inverse      (gray dotted)  -- subspace iteration with the EXACT
%     A^{-1} (chol decomposition), the ideal-operator floor: it shows the
%     best capture any approximate inverse could reach at the same q, so the
%     gap AMG-curve -> exact-inverse curve is the price of approximating
%     A^{-1} by one V-cycle;
%   exact_inverse_plain (light gray dotted) -- the SAME exact inverse but
%     WITHOUT per-step re-orthonormalization (plain iteration, as in
%     subspace_capture/run_inverse_subspace_iter.m): demonstrates the
%     roundoff rank collapse of an un-reorthogonalized block (r_comp/r_defl
%     shrink, capture stalls), which is why those curves truncate past
%     q ~ 4 while the orth'd twin keeps converging.
% (Cost data -- setup time, cumulative apply/orth time, work units -- is
% recorded in the CSVs, not plotted.  The exact-inverse rows carry NaN
% setup/work-unit proxies: its dense-factor cost is not comparable to the
% V-cycle work model, only its measured apply time is meaningful.)
%
% Usage:
%   cd amg_subspace
%   run_amg_subspace_capture

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);                                 % src.* packages
addpath(fullfile(repoRoot, 'subspace_capture'));   % capture + kappa metrics
addpath(thisFileDir);                              % local AMG factory

%% --- Configuration ---------------------------------------------------------
% true = fast end-to-end check (small mesh, 4 configs).  Pre-set `smoke` in
% the workspace (e.g. `smoke = true; run_amg_subspace_capture`) to override.
if ~exist('smoke', 'var'), smoke = false; end

if smoke
    h0        = 0.15;
    k         = 40;
    iters     = [0 1 2 4];
    minCoarse = 100;
    outDir    = fullfile(thisFileDir, 'output_smoke');
else
    h0        = 0.05;
    k         = 200;
    iters     = [0 1 2 3 4 5 6 8 10 12 16 20];
    minCoarse = 800;
    outDir    = fullfile(thisFileDir, 'output');
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

%% --- Ground truth: smallest k eigenvectors of A ----------------------------
dA = decomposition(A, 'chol');
[V_true, lam_cut] = load_or_compute_eigs_A(cacheDir, A, dA, k, h0);
fprintf('V_true (smallest %d eigvecs of A): %d x %d  (lam_cut=%.4e)\n', ...
        k, size(V_true,1), size(V_true,2), lam_cut);

%% --- kappa_exact: best-achievable deflated condition number ----------------
Afun    = @(x) A * x;
Ainvfun = @(x) dA \ x;
condOpts    = struct('eigs_tol', 1e-8, 'eigs_maxit', 5000, 'W_is_orth', true);
% Raw-block variant (no W_is_orth): the un-reorthogonalized 'plain' config
% passes a non-orthonormal V, whose internal pivoted QR records the rank.
condOptsRaw = struct('eigs_tol', 1e-8, 'eigs_maxit', 5000);
exactCond = deflated_cond_two_level(V_true, Afun, Ainvfun, lam_cut, n, condOpts);
if ~exactCond.ok
    error('run_amg_subspace_capture:kappaExactFailed', ...
          'kappa_exact computation failed: %s', exactCond.err);
end
kappa_exact = exactCond.kappa;
lam_max_A   = exactCond.lam_max;
fprintf('kappa_exact = %.6e  (analytic lam_max_A/lam_cut = %.6e)\n', ...
        kappa_exact, lam_max_A / lam_cut);

%% --- Shared Gaussian starting block ---------------------------------------
rng(seed);
V0 = randn(n, m);

%% --- Sweep over AMG configurations ----------------------------------------
ops = struct('Afun', Afun, 'Ainvfun', Ainvfun, 'lam_cut', lam_cut, ...
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

rows = [];
for ic = 1:numel(configs)
    cfg = configs(ic);
    fprintf('\n[%d/%d] %s  (proj=%s pre=%d post=%d coarse=%s)\n', ...
            ic, numel(configs), cfg.name, cfg.projector, ...
            cfg.preSmooth, cfg.postSmooth, cfg.coarseSolve);

    if strcmp(cfg.projector, 'exact')
        % Ideal-operator reference: exact inverse iteration via the chol
        % decomposition already built for the ground-truth eigs.  The AMG
        % cost model does not apply (NaN work proxies); measured apply time
        % is still recorded by the sweep.
        Mfun  = @(X) dA \ X;
        ainfo = struct('projector', 'exact', 'sjltNnzPerCol', NaN, ...
                       'levels', struct('n', n, 'nnzA', nnz(A), 'nnzP', 0), ...
                       'nLevels', 1, 'coarseN', NaN, 'coarseNnz', NaN, ...
                       'coarseType', 'exact', 'setupTime', NaN, ...
                       'workPerApply', NaN, 'workUnits', NaN);
        fprintf('  exact inverse A^{-1} (chol decomposition reference)\n');
    else
        % Reseed per config: sjlt draws become reproducible AND distinct.
        rng(seed + ic);
        % Per-config aggregate size (section E sweeps it; NaN = global default).
        effMaxAgg = maxAggSize;
        if isfinite(cfg.maxAggSize), effMaxAgg = cfg.maxAggSize; end
        [Mfun, ainfo] = make_amg_prec_ablate(A, ...
            'maxLevels', 2, 'minCoarseSize', minCoarse, ...
            'theta', theta, 'maxAggSize', effMaxAgg, ...
            'omegaInterp', 0, 'omegaSmooth', omegaSmooth, ...
            'preSmooth', cfg.preSmooth, 'postSmooth', cfg.postSmooth, ...
            'coarseSolve', cfg.coarseSolve, ...
            'coarseJacobiSweeps', cfg.coarseJacobiSweeps, ...
            'coarsePcgTol', cfg.coarsePcgTol, ...
            'coarsePcgMaxit', cfg.coarsePcgMaxit, ...
            'projector', cfg.projector, ...
            'sjltNc', cfg.sjltNc, 'sjltNnzPerCol', cfg.sjltNnzPerCol, ...
            'fineSmootherL', L, 'fineSmootherLt', Lt);

        % Section E labels by the REALIZED coarse size (maxAggSize -> nc is
        % emergent), so the legend reads the achieved nc, not the knob.
        if strcmp(cfg.section, 'E_sa_coarse_size')
            configs(ic).label = sprintf('SA nc=%d', ainfo.coarseN);
        end

        fprintf('  levels %s  coarse=%s  setup=%.2fs  WU/apply=%.2f\n', ...
                mat2str([ainfo.levels.n]), ainfo.coarseType, ...
                ainfo.setupTime, ainfo.workUnits);
    end
    if cfg.preSmooth == 0 && cfg.postSmooth == 0 && ainfo.coarseN < m
        warning('run_amg_subspace_capture:rankLimited', ...
                ['config %s: pure coarse-grid correction with coarseN=%d < ', ...
                 'm=%d -- captured subspace is rank-limited.'], ...
                cfg.name, ainfo.coarseN, m);
    end

    cfgRows = run_one_config(cfg, Mfun, ainfo, V0, V_true, ops);
    rows = [rows; cfgRows];                                       %#ok<AGROW>
end

%% --- Save ------------------------------------------------------------------
meta = struct('n', n, 'k', k, 'm', m, 'iters', iters, ...
              'h0', h0, 'contrast', contrast, 't_snap', t_snap, 'dt', dt, ...
              'seed', seed, 'theta', theta, 'maxAggSize', maxAggSize, ...
              'omegaSmooth', omegaSmooth, 'minCoarseSize', minCoarse, ...
              'maxLevels', 2, 'lam_cut', lam_cut, 'lam_max_A', lam_max_A, ...
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
%   Naming:  <proj>_pre<i>_post<j>_<coarse>[_nc<size>]
%     proj   : sa | sjlt1 | sjlt4    (digit = s, nnz per fine row)
%     nc     : zero-padded coarse size for sjlt sweep configs; 'ncmatch' means
%              "match the SA baseline's realized nc" (resolved at runtime).
%   section : which ablation the config belongs to / which plot folder it is
%             drawn in.  'baseline' = sa_pre1_post1_chol, drawn (black, solid)
%             in EVERY section.
%   Visuals: solid = SA, dashed = SJLT, dash-dot = cross configs.
    C = struct('name', {}, 'preSmooth', {}, 'postSmooth', {}, ...
               'coarseSolve', {}, 'coarseJacobiSweeps', {}, ...
               'coarsePcgTol', {}, 'coarsePcgMaxit', {}, ...
               'projector', {}, 'sjltNc', {}, 'sjltNnzPerCol', {}, ...
               'maxAggSize', {}, ...
               'section', {}, 'label', {}, 'color', {}, 'style', {});

    add = @(C, name, pre, post, cs, nu, tol, mit, proj, nc, s, sect, col, st) ...
        [C, struct('name', name, 'preSmooth', pre, 'postSmooth', post, ...
                   'coarseSolve', cs, 'coarseJacobiSweeps', nu, ...
                   'coarsePcgTol', tol, 'coarsePcgMaxit', mit, ...
                   'projector', proj, 'sjltNc', nc, 'sjltNnzPerCol', s, ...
                   'maxAggSize', NaN, ...
                   'section', sect, 'label', name, 'color', col, 'style', st)];

    % --- references (drawn in every section) -------------------------------
    % exact_inverse sits outside the <proj>_... naming scheme: it is not an
    % AMG config but the ideal-operator floor V <- orth(A^{-1} V).
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
    % --- E_sa_coarse_size: single-pass SA coarse-size sweep via maxAggSize.
    %     One aggregation pass floors at nc ~ n/4.6 (nc_sa, the baseline drawn
    %     in every section); SMALLER maxAggSize -> smaller aggregates -> LARGER
    %     nc, so this sweeps the achievable single-pass range [nc_sa, ~n/1.8]
    %     upward from the baseline.  SA analog of section D; realized nc is read
    %     back into the legend label at run time.  warm -> cool with growing nc.
    masSweep  = [6 4 3 2];
    masColors = [0.99 0.75 0.15;  0.85 0.45 0.20;  0.55 0.30 0.60;  0.20 0.45 0.80];
    for iv = 1:numel(masSweep)
        C = add(C, sprintf('sa_pre1_post1_chol_mas%02d', masSweep(iv)), ...
                1, 1, 'chol', NaN, NaN, NaN, ...
                'sa', NaN, NaN, 'E_sa_coarse_size', masColors(iv,:), '-');
        C(end).maxAggSize = masSweep(iv);
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
                         'sjlt4_pre1_post1_chol_ncmatch', ...
                         'sa_pre1_post1_chol_mas02'});
        C = C(keep);
    end
    configs = C;
end

function rows = run_one_config(cfg, Mfun, ainfo, V0, V_true, ops)
%RUN_ONE_CONFIG  One incremental q-sweep: V <- orth(Mfun(V)), metrics at each
%   recorded q.  cfg.reorth = true re-orthonormalizes EVERY step: M's
%   spectrum decays like 1/lambda(A), so an unorthogonalized block loses the
%   trailing directions to roundoff well before q = 20.  cfg.reorth = false
%   skips the in-loop orth (plain iteration, as in
%   run_inverse_subspace_iter.m); the metrics then orthonormalize a
%   rank-truncating copy, so r_comp/r_defl record the collapse.
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
            warning('run_amg_subspace_capture:captureFailed', ...
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
%   Delegates to amg_bench_snapshot, the one shared definition of this test
%   problem, so all three amg_subspace drivers assemble a bit-identical A --
%   which is what keeps the (h0, k)-keyed eigenvector caches valid across them.
%   Three outputs, so the RHS is not built here: this driver has no solver.
    [A, L, msh] = amg_bench_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
end

function [V_true, lam_cut] = load_or_compute_eigs_A(cacheDir, A, dA, k, h0)
%LOAD_OR_COMPUTE_EIGS_A  Smallest k+1 eigenpairs of A itself (cached).
%   Cache key includes h0 and k so smoke and full runs never collide.
%   NOTE: the old subspace_capture caches hold eigenpairs of Tsym, not A --
%   they cannot be reused here.
    cachePath = fullfile(cacheDir, sprintf('eigsA_h%g_k%d.mat', h0, k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_true  = S.V(:, 1:k);
        lam_cut = S.D(k + 1);
        fprintf('Loaded cached %s\n', cachePath);
        return;
    end
    fprintf('Computing smallest %d eigenpairs of A (one-time)...\n', k + 1);
    n  = size(A, 1);
    t0 = tic;
    % Name-value options, NOT the legacy opts struct: eigs silently ignores
    % struct fields with these capitalized names.  With a function handle and
    % 'smallestabs', eigs expects the handle to apply A^{-1} x.
    [Vraw, Dmat] = eigs(@(x) dA \ x, n, k + 1, 'smallestabs', ...
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
    sections = {'A_smoothing', 'B_inner_solve', 'C_projector', ...
                'D_coarse_size', 'E_sa_coarse_size'};
    secTitles = { ...
        'A: pre/post smoothing sweeps (SA projector, exact coarse solve)', ...
        'B: coarse inner solve (SA projector, at/around pre=post=1)', ...
        'C: projector -- SA aggregation vs SJLT sketch at matched nc', ...
        'D: SJLT (s=4) coarse-size sweep', ...
        'E: SA coarse-size sweep (single-pass, varied maxAggSize)'};

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
        'title',  {'eigenspace error', 'angle capture fraction', ...
                   'deflated condition-number ratio'}, ...
        'tag',    {'eigspace_err2_vs_q', 'angle_capture_fraction_vs_q', ...
                   'kappa_ratio_vs_q'});

    for ip = 1:numel(specs)
        fig = figure('Visible', 'off', 'Units', 'inches', ...
                     'Position', [0 0 5.4 4.3], 'Color', 'w');
        ax = axes(fig);
        [h, keep, labels] = draw_panel(ax, rows, configs, specs(ip));
        lgd = legend(ax, h(keep), labels, 'Location', 'southoutside', ...
                     'Box', 'on', 'EdgeColor', [0.65 0.65 0.65], ...
                     'Color', 'white', 'FontSize', 6.5, 'NumColumns', 3, ...
                     'Interpreter', 'none');
        lgd.ItemTokenSize = [14, 6];
        outfile = fullfile(out_dir, [specs(ip).tag '.pdf']);
        exportgraphics(fig, outfile, 'ContentType', 'vector');
        close(fig);
        fprintf('Wrote %s\n', outfile);
    end

    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 16.5 4.6], 'Color', 'w');
    tl = tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    legHandles = gobjects(0);
    legLabels  = {};
    for ip = 1:numel(specs)
        [h, keep, labels] = draw_panel(nexttile(tl), rows, configs, specs(ip));
        if ip == 1
            legHandles = h(keep);
            legLabels  = labels;
        end
    end
    title(tl, sec_title, 'FontWeight', 'bold', 'FontSize', 12);
    lgd = legend(legHandles, legLabels, 'Box', 'on', ...
                 'EdgeColor', [0.65 0.65 0.65], 'Color', 'white', ...
                 'FontSize', 8, 'Interpreter', 'none', ...
                 'NumColumns', ceil(numel(legLabels) / 2));
    lgd.Layout.Tile = 'south';
    lgd.ItemTokenSize = [18, 8];
    outfile = fullfile(out_dir, 'aggregate_1x3.png');
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
    fprintf('Wrote %s\n', outfile);
end

function [handles, keep, labels] = draw_panel(ax, rows, configs, spec)
%DRAW_PANEL  One metric-vs-x panel with a series per config.  Returns the line
%   handles, the keep mask, and the labels so the CALLER places the legend
%   (outside the axes), keeping it off the data.  Each series gets a distinct
%   marker so overlapping same-color/same-style curves stay separable.
    markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', '*'};
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
        mk = markers{mod(ic - 1, numel(markers)) + 1};
        handles(ic) = plot(ax, xs, ys, [cfg.style mk], ...
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

    labels = {configs(keep).label};
end

