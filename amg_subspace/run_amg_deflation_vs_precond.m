% RUN_AMG_DEFLATION_VS_PRECOND  Is an AMG-sketched deflation basis worth more
% than the AMG V-cycle itself?
%
% run_amg_subspace_capture.m established that an AMG V-cycle, used to drive
% subspace iteration, captures the smallest-eigenvalue eigenspace of the sphere
% heat-step operator A very well.  What it never measured is the spectrum of
% the operator a Krylov method actually sees.  This study does, and asks:
%
%   given that the V-cycle and a basis sketched FROM that V-cycle carry the
%   same subspace information, which one turns it into a better-conditioned
%   system -- and does a cheap, weak AMG stay a good subspace generator after
%   it has stopped being a good preconditioner?
%
% The two mechanisms differ structurally, which is why the two ends of the
% spectrum are reported separately rather than only as their ratio:
%   deflation  P = (I-VV') + tau V(V'AV)^{-1}V'  moves span(V) to exactly tau
%              and leaves everything else -- lam_max included -- untouched.
%              With an exact eigenbasis, kappa(PA) = lam_max(A)/lam_{r+1}:
%              the bottom is fixed perfectly, the top not at all.
%   AMG        M compresses BOTH ends, but only approximately, and its quality
%              at the bottom is exactly the capture quality already measured.
%
% FAIRNESS.  Every competing arm uses only sketch-derived information: a
% Gaussian block, q applications of that config's own V-cycle, A matvecs, and
% small dense operations on the result.  That includes the deflation shift tau,
% which is the top Ritz value of A on span(V) (amg_sketch_tau) rather than a
% cached eigenvalue -- handing deflation an exact lam_{r+1} would give it
% spectral information the AMG arm never gets.  Exact eigenvectors appear in
% exactly two places, both labelled and never as an arm's input: the defl_exact
% reference floor, and the capture metric.
%
% Arms (amg_deflation_arms.m), all preconditioning the SAME system A x = b:
%   pcg_plain       no preconditioner -- the reference for both spectral ends
%   pcg_ichol       ichol(A,'nofill')
%   amg_direct      the V-cycle as the preconditioner            <- to beat
%   defl_amg        deflation with the sketched basis, no AMG in the solve
%   defl_amg_ichol  ichol smoothing + the sketched coarse space (C_tau form)
%   ctau_amg        V-cycle AND deflation together (C_tau form)
%   defl_exact      deflation with exact eigenvectors -- reference floor,
%                   drawn at BOTH d = k_target and d = m_sketch so no
%                   dimension-mismatched comparison can be read off a plot
%
% Metrics.  Primary: lam_min, lam_max and kappa of the preconditioned operator,
% via precond_spectrum (symmetric Lanczos on H = R M R', R = chol(A)) -- one
% routine for every arm, so the numbers are comparable.  Secondary: PCG
% iterations to tolerance, the empirical companion.  Capture quality of each
% sketch is recorded alongside via subspace_capture_directed, so the money plot
% can put subspace quality on one axis and conditioning on the other.
%
% COST CAVEAT.  Iteration counts here exclude the q*m_sketch V-cycle applies
% spent building the basis, and an AMG-preconditioned iteration costs about
% work_units_per_amg_apply times more than a deflation-only one.  Cost columns
% are recorded in the CSV but deliberately not analyzed: this study is about
% conditioning, not about amortization.
%
% Outputs (amg_subspace/output_defl/):
%   results.mat / results.csv   one row per (config, k_target, q, arm)
%   money.csv                   one row per (config, k_target, q), arms as columns
%   spectra.mat                 Ritz tails for the spectrum panels
%   *.pdf / aggregate_2x3.png   see make_amg_defl_plots.m
%
% Usage:
%   cd amg_subspace
%   run_amg_deflation_vs_precond
%   smoke = true; run_amg_deflation_vs_precond      % fast end-to-end check

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);                                 % src.* packages
addpath(fullfile(repoRoot, 'subspace_capture'));   % capture + kappa metrics
addpath(thisFileDir);                              % local AMG factory + helpers

%% --- Configuration ---------------------------------------------------------
if ~exist('smoke', 'var'), smoke = false; end

rho = 2;                     % Gaussian oversampling factor: m_sketch = rho*k_target

if smoke
    h0        = 0.15;
    k         = 40;
    ktargets  = [10 20];
    qs        = [1 3];
    minCoarse = 100;
    lanczos_m = 150;
    pcg_maxit = 200;
    outDir    = fullfile(thisFileDir, 'output_defl_smoke');
    cacheDir  = fullfile(thisFileDir, 'output_smoke', 'cache');
else
    h0        = 0.05;
    k         = 200;
    ktargets  = [25 50 75 100];
    qs        = [1 2 4 8];
    minCoarse = 800;
    lanczos_m = 300;
    pcg_maxit = 400;
    outDir    = fullfile(thisFileDir, 'output_defl');
    cacheDir  = fullfile(thisFileDir, 'output', 'cache');
end

% The defl_exact floors need d+1 <= k+1 cached eigenvectors at d = rho*k_target,
% which is what caps k_target at k/rho.  Reusing the sibling's cache instead of
% recomputing a wider one is worth the cap.
assert(rho * max(ktargets) <= k, ...
       'rho*max(ktargets) = %d exceeds the cached eigenvector count k = %d.', ...
       rho * max(ktargets), k);

if ~isfolder(outDir),   mkdir(outDir);   end
if ~isfolder(cacheDir), mkdir(cacheDir); end

% Snapshot + RHS (identical to run_amg_subspace_capture.m / run_krylov_capture.m).
contrast    = 60;
t_snap      = 0;
dt          = 1;
Tmax        = 100;
mesh_method = 'pdetoolbox';
Kmodes      = 50;
sigma       = 1;

% Fixed AMG structure, matching the capture study so its curves line up.
theta       = 0.05;
maxAggSize  = 16;
omegaSmooth = 2/3;
seed        = 1;
pcg_tol     = 1e-8;

configs = make_amg_configs(smoke);
arms    = amg_deflation_arms();

%% --- Build snapshot A, ichol factor L, RHS b -------------------------------
fprintf('\n--- Building sphere snapshot (h0=%.4g, %s) ---\n', h0, mesh_method);
[A, L, msh, b] = amg_bench_snapshot(h0, contrast, t_snap, dt, Tmax, ...
                                    mesh_method, Kmodes, sigma, seed);
n  = msh.numIN;
Lt = L';
fprintf('A: %d x %d, nnz=%d, sym=%d   nnz(L)=%d   ||b||=%.4e\n', ...
        size(A,1), size(A,2), nnz(A), issymmetric(A), nnz(L), norm(b));

Afun  = @(X) A * X;
Rchol = chol(A);                       % upper, A = R'R -- drives precond_spectrum
fprintf('chol(A): nnz=%d\n', nnz(Rchol));

%% --- Ground truth (reference only, never an arm's input) -------------------
dA = decomposition(A, 'chol');
[V_true_all, D_all] = load_or_compute_eigs_full(cacheDir, A, dA, k, h0);
fprintf('Cached eigenpairs: %d vectors, lam_1=%.4e ... lam_%d=%.4e\n', ...
        size(V_true_all,2), D_all(1), numel(D_all), D_all(end));

specOpts = struct('m', lanczos_m, 'n_tail', min(200, n));
captOpts = struct('true_is_orth', true, 'comp_is_orth', true);

ctx = struct('A', A, 'Afun', Afun, 'L', L, 'Lt', Lt, 'Mamg', [], ...
             'n', n, 'M_is_sym', true);

rows    = [];
spectra = [];

%% --- Reference arms (no AMG config involved) -------------------------------
fprintf('\n--- Reference arms ---\n');

% pcg_plain first: its extremes normalize every lam_*_ratio column.
[rowP, specP] = run_arm(arms(strcmp({arms.id}, 'pcg_plain')), ctx, [], NaN, ...
                        blank_row(), A, b, pcg_tol, pcg_maxit, Rchol, n, specOpts);
lam_min_ref = rowP.lam_min;
lam_max_ref = rowP.lam_max;
fprintf('  unpreconditioned: lam_min=%.4e lam_max=%.4e kappa=%.4e iters=%d\n', ...
        rowP.lam_min, rowP.lam_max, rowP.kappa, rowP.iters);

refRows  = [];
refSpecs = [];
for ia = 1:numel(arms)
    % pcg_plain was already run above to establish the normalizing extremes;
    % running it again here would duplicate its row.
    if ~arms(ia).is_reference || arms(ia).needs_V || ...
       strcmp(arms(ia).id, 'pcg_plain')
        continue;
    end
    base = blank_row();
    base.config = 'reference';  base.section = 'reference';
    [r, sp] = run_arm(arms(ia), ctx, [], NaN, base, A, b, pcg_tol, pcg_maxit, ...
                      Rchol, n, specOpts);
    refRows  = [refRows;  r];    %#ok<AGROW>
    refSpecs = [refSpecs; sp];   %#ok<AGROW>
    fprintf('  %-12s: lam_min=%.4e lam_max=%.4e kappa=%.4e iters=%d\n', ...
            r.arm, r.lam_min, r.lam_max, r.kappa, r.iters);
end

% defl_exact at every deflation dimension that appears anywhere in the sweep:
% d = k_target (what the sketch was aiming at) and d = m_sketch (what it
% actually spans).  Both floors get drawn; conflating them is the easiest way
% to publish a 2k-vs-k comparison by accident.
exact_dims   = unique([ktargets(:); rho * ktargets(:)])';
kappa_exact  = containers.Map('KeyType', 'double', 'ValueType', 'double');
armExact     = arms(strcmp({arms.id}, 'defl_exact'));
for d = exact_dims
    Vd    = V_true_all(:, 1:d);
    tau_d = D_all(d + 1);
    base  = blank_row();
    base.config = 'reference';  base.section = 'reference';
    base.r_defl = d;  base.m_sketch = d;  base.k_target = d;
    base.tau_exact_ref = tau_d;  base.tau_ritz_gap = 1;
    [r, sp] = run_arm(armExact, ctx, Vd, tau_d, base, A, b, pcg_tol, pcg_maxit, ...
                      Rchol, n, specOpts);
    r.arm = sprintf('defl_exact_d%d', d);
    sp.arm = r.arm;
    refRows  = [refRows;  r];    %#ok<AGROW>
    refSpecs = [refSpecs; sp];   %#ok<AGROW>
    kappa_exact(d) = r.kappa;
    fprintf('  defl_exact d=%-4d: lam_min=%.4e (tau=%.4e) lam_max=%.4e kappa=%.4e iters=%d\n', ...
            d, r.lam_min, tau_d, r.lam_max, r.kappa, r.iters);
end

rows    = [rows;    rowP; refRows];
spectra = [spectra; specP; refSpecs];

%% --- Resolve the matched coarse size for the '_ncmatch' sjlt configs -------
[~, saInfo] = make_amg_prec_ablate(A, ...
    'maxLevels', 2, 'minCoarseSize', minCoarse, ...
    'theta', theta, 'maxAggSize', maxAggSize, ...
    'omegaInterp', 0, 'omegaSmooth', omegaSmooth, ...
    'fineSmootherL', L, 'fineSmootherLt', Lt);
nc_sa = saInfo.coarseN;
fprintf('\nSA realized coarse size nc_sa = %d (used by the _ncmatch configs)\n', nc_sa);
for ic = 1:numel(configs)
    if strcmp(configs(ic).projector, 'sjlt') && isnan(configs(ic).sjltNc)
        configs(ic).sjltNc = nc_sa;
        configs(ic).label  = sprintf('%s (nc=%d)', configs(ic).label, nc_sa);
    end
end

%% --- Sweep: AMG config x sketch width x subspace iterations ----------------
for ic = 1:numel(configs)
    cfg = configs(ic);
    fprintf('\n[%d/%d] %s  (proj=%s pre=%d post=%d coarse=%s)\n', ...
            ic, numel(configs), cfg.name, cfg.projector, ...
            cfg.preSmooth, cfg.postSmooth, cfg.coarseSolve);

    rng(seed + ic);                     % sjlt draws: reproducible AND distinct
    effMaxAgg = maxAggSize;
    if isfinite(cfg.maxAggSize), effMaxAgg = cfg.maxAggSize; end

    t0 = tic;
    [Mamg, ainfo] = make_amg_prec_ablate(A, ...
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
    setup_sec_amg = toc(t0);
    fprintf('  levels %s  coarse=%s  setup=%.2fs  WU/apply=%.2f  M_is_sym=%d\n', ...
            mat2str([ainfo.levels.n]), ainfo.coarseType, setup_sec_amg, ...
            ainfo.workUnits, cfg.M_is_sym);

    ctx.Mamg     = Mamg;
    ctx.M_is_sym = cfg.M_is_sym;

    base0 = blank_row();
    base0.config     = cfg.name;
    base0.section    = cfg.section;
    base0.projector  = cfg.projector;
    base0.preSmooth  = cfg.preSmooth;
    base0.postSmooth = cfg.postSmooth;
    base0.coarseSolve = cfg.coarseSolve;
    base0.M_is_sym   = cfg.M_is_sym;
    base0.coarseN    = ainfo.coarseN;
    base0.rho        = rho;
    base0.setup_sec_amg = setup_sec_amg;
    base0.work_units_per_amg_apply = ainfo.workUnits;

    % amg_direct is subspace-independent: measure it once per config.
    armDirect = arms(strcmp({arms.id}, 'amg_direct'));
    [rD, spD] = run_arm(armDirect, ctx, [], NaN, base0, A, b, pcg_tol, ...
                        pcg_maxit, Rchol, n, specOpts);
    rD  = finish_ratios(rD, lam_min_ref, lam_max_ref, kappa_exact);
    rows    = [rows;    rD];   %#ok<AGROW>
    spectra = [spectra; spD];  %#ok<AGROW>
    if rD.ok
        fprintf('    amg_direct: lam_min=%.4e lam_max=%.4e kappa=%.4e iters=%d\n', ...
                rD.lam_min, rD.lam_max, rD.kappa, rD.iters);
    else
        fprintf('    amg_direct: SKIPPED (%s)\n', rD.skip_reason);
    end

    for kt = ktargets
        m_sk = rho * kt;
        sketchState = [];
        for q = sort(qs, 'ascend')
            % Incremental continuation: sweeping q upward costs max(qs)
            % applies in total, not sum(qs).
            [Vd, sinfo] = amg_sketch_basis(Mamg, n, m_sk, q, seed, sketchState);
            sketchState = sinfo;

            if sinfo.rank_limited
                warning('run_amg_deflation_vs_precond:rankLimited', ...
                        ['config %s: sketch of width %d collapsed to rank %d ', ...
                         '(coarseN=%d) -- the V-cycle cannot span more.'], ...
                        cfg.name, m_sk, sinfo.r_defl, ainfo.coarseN);
            end

            [tau, tinfo] = amg_sketch_tau(Vd, Afun);
            capt = subspace_capture_directed(V_true_all(:, 1:kt), Vd, [], captOpts);

            base = base0;
            base.k_target      = kt;
            base.m_sketch      = m_sk;
            base.r_defl        = sinfo.r_defl;
            base.q             = q;
            base.setup_sec_V   = sinfo.time_seconds;
            base.tau           = tau;
            base.tau_exact_ref = D_all(min(sinfo.r_defl + 1, numel(D_all)));
            base.tau_ritz_gap  = tau / base.tau_exact_ref;
            base.eigspace_err_2 = capt.eigspace_err_2;
            % Same normalization the capture study uses: the fraction of true
            % directions captured to within sin(theta) < 1%.
            base.angle_capture_frac_1pct = ...
                capt.n_angle_below_1pct / max(capt.r_true, 1);
            base.r_comp        = capt.r_comp;

            for ia = 1:numel(arms)
                a = arms(ia);
                if ~a.needs_V || a.is_reference, continue; end

                if ~tinfo.ok
                    r = base;  r.arm = a.id;  r.ok = false;
                    r.skip_reason = tinfo.err;
                    rows = [rows; r];   %#ok<AGROW>
                    continue;
                end

                [r, sp] = run_arm(a, ctx, Vd, tau, base, A, b, pcg_tol, ...
                                  pcg_maxit, Rchol, n, specOpts);
                r = finish_ratios(r, lam_min_ref, lam_max_ref, kappa_exact);
                rows    = [rows;    r];   %#ok<AGROW>
                spectra = [spectra; sp];  %#ok<AGROW>
            end

            % --- Shift side-sweep, at the reference sketch width only ------
            % tau sets lam_min of the deflated operator directly, so without
            % this the sketch-only SHIFT and the sketched SUBSPACE cannot be
            % told apart: a config whose kappa is poor only because tau is
            % badly placed would be misreported as having a poor subspace.
            % Both alternatives are REFERENCE ONLY -- 'exact' consumes a
            % cached eigenvalue, so it is not a fair competitor; it is here to
            % show what the sketch-only shift costs (or gains) against the
            % value that is optimal for an EXACT eigenbasis.
            if kt == max(ktargets) && tinfo.ok
                altTaus = { 'exact',   base.tau_exact_ref, false; ...
                            'lam_max', lam_max_ref,        true };
                armDefl = arms(strcmp({arms.id}, 'defl_amg'));
                for it = 1:size(altTaus, 1)
                    b2              = base;
                    b2.tau_mode     = altTaus{it, 1};
                    b2.tau          = altTaus{it, 2};
                    b2.tau_ritz_gap = b2.tau / base.tau_exact_ref;
                    a2              = armDefl;
                    a2.sketch_only  = altTaus{it, 3};
                    [r2, sp2] = run_arm(a2, ctx, Vd, b2.tau, b2, A, b, ...
                                        pcg_tol, pcg_maxit, Rchol, n, specOpts);
                    % Distinct arm id: the default sketch-shift row must stay
                    % the unique 'defl_amg' entry the plots and money.csv pick.
                    r2.arm  = sprintf('defl_amg_tau_%s', b2.tau_mode);
                    sp2.arm = r2.arm;
                    r2 = finish_ratios(r2, lam_min_ref, lam_max_ref, kappa_exact);
                    rows    = [rows;    r2];    %#ok<AGROW>
                    spectra = [spectra; sp2];   %#ok<AGROW>
                end
            end

            sel = arrayfun(@(r) r.k_target == kt && r.q == q && ...
                                strcmp(r.config, cfg.name), rows);
            fprintf(['    m_sk=%3d q=%d  r=%3d  tau=%.3e (gap %.3f)  ', ...
                     'capture_err=%.3e  kappa: %s\n'], ...
                    m_sk, q, sinfo.r_defl, tau, base.tau_ritz_gap, ...
                    capt.eigspace_err_2, kappa_summary(rows(sel)));
        end
    end
end

%% --- Save ------------------------------------------------------------------
meta = struct('n', n, 'k', k, 'rho', rho, 'ktargets', ktargets, 'qs', qs, ...
              'h0', h0, 'contrast', contrast, 't_snap', t_snap, 'dt', dt, ...
              'seed', seed, 'theta', theta, 'maxAggSize', maxAggSize, ...
              'omegaSmooth', omegaSmooth, 'minCoarseSize', minCoarse, ...
              'maxLevels', 2, 'lanczos_m', lanczos_m, 'pcg_tol', pcg_tol, ...
              'pcg_maxit', pcg_maxit, 'lam_min_ref', lam_min_ref, ...
              'lam_max_ref', lam_max_ref, 'nc_sa', nc_sa, 'smoke', smoke, ...
              'Kmodes', Kmodes, 'exact_dims', exact_dims, ...
              'config_names', {{configs.name}});

save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');
save(fullfile(outDir, 'spectra.mat'), 'spectra', 'meta', '-v7');
write_defl_results_csv(fullfile(outDir, 'results.csv'), rows);
write_money_csv(fullfile(outDir, 'money.csv'), rows, configs, ktargets, qs, rho);
fprintf('\nresults.mat / results.csv / money.csv / spectra.mat written to:\n  %s\n', outDir);

%% --- Lanczos drift check ---------------------------------------------------
% lam_min is the fragile end of a Ritz spectrum.  Re-measure the baseline
% config's amg_direct arm at half the Lanczos budget: if kappa moves by more
% than a percent, the reported condition numbers are budget-limited and
% lanczos_m should be raised rather than the numbers reported.
driftRow = rows(find(strcmp({rows.arm}, 'amg_direct') & [rows.ok], 1));
if ~isempty(driftRow)
    icDrift = find(strcmp({configs.name}, driftRow.config), 1);
    rng(seed + icDrift);
    Mdrift = make_amg_prec_ablate(A, 'maxLevels', 2, 'minCoarseSize', minCoarse, ...
        'theta', theta, 'maxAggSize', maxAggSize, 'omegaInterp', 0, ...
        'omegaSmooth', omegaSmooth, 'preSmooth', configs(icDrift).preSmooth, ...
        'postSmooth', configs(icDrift).postSmooth, ...
        'coarseSolve', configs(icDrift).coarseSolve, ...
        'fineSmootherL', L, 'fineSmootherLt', Lt);
    half = precond_spectrum(Mdrift, Rchol, n, ...
                            struct('m', max(round(lanczos_m/2), 10), 'n_tail', 10));
    meta.lanczos_drift = abs(half.kappa - driftRow.kappa) / driftRow.kappa;
    fprintf(['Lanczos drift check (%s, amg_direct): kappa %.4e at m=%d vs %.4e ', ...
             'at m=%d -- rel %.3g\n'], driftRow.config, half.kappa, half.m_used, ...
            driftRow.kappa, lanczos_m, meta.lanczos_drift);
    if meta.lanczos_drift > 1e-2
        warning('run_amg_deflation_vs_precond:lanczosDrift', ...
                ['kappa moved %.3g between m=%d and m=%d: the reported condition ', ...
                 'numbers are Lanczos-budget-limited, raise lanczos_m.'], ...
                meta.lanczos_drift, half.m_used, lanczos_m);
    end
    save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');
end

%% --- Plots -----------------------------------------------------------------
fprintf('\n--- Rendering plots ---\n');
make_amg_defl_plots(rows, spectra, configs, meta, outDir);
fprintf('\nDone.\n');

%% =========================================================================
%% Local helpers
%% =========================================================================
function configs = make_amg_configs(smoke)
%MAKE_AMG_CONFIGS  Strong -> cheap/weak spine, drawn from the ablation grid of
%   run_amg_subspace_capture.m (same field names, same knobs).  The point here
%   is spanning preconditioner quality, not re-running the full 25-config
%   ablation: each config must be a materially different answer to "how good a
%   preconditioner is this V-cycle", so that "how good a subspace does it
%   sketch" can be plotted against it.
    c = @(name, section, label, pre, post, coarse, jac, proj, nc, s, mas) struct( ...
        'name', name, 'section', section, 'label', label, ...
        'preSmooth', pre, 'postSmooth', post, ...
        'coarseSolve', coarse, 'coarseJacobiSweeps', jac, ...
        'coarsePcgTol', 1e-2, 'coarsePcgMaxit', 50, ...
        'projector', proj, 'sjltNc', nc, 'sjltNnzPerCol', s, ...
        'maxAggSize', mas, 'M_is_sym', pre == post);

    configs = [ ...
        c('sa_pre1_post1_chol', 'baseline',   'SA pre1/post1 (baseline)', 1, 1, 'chol',   2, 'sa',   [],  4, NaN); ...
        c('sa_pre2_post2_chol', 'A_smoothing','SA pre2/post2',            2, 2, 'chol',   2, 'sa',   [],  4, NaN); ...
        c('sa_pre0_post0_chol', 'A_smoothing','SA pure CGC (pre0/post0)', 0, 0, 'chol',   2, 'sa',   [],  4, NaN); ...
        c('sa_pre1_post1_jac2', 'B_inner',    'SA coarse jacobi(2)',      1, 1, 'jacobi', 2, 'sa',   [],  4, NaN); ...
        c('sjlt4_ncmatch',      'C_projector','SJLT s=4, nc matched',     1, 1, 'chol',   2, 'sjlt', NaN, 4, NaN); ...
        c('sjlt4_nc0200',       'D_coarse',   'SJLT s=4, nc=200',         1, 1, 'chol',   2, 'sjlt', 200, 4, NaN); ...
        c('sa_pre1_post0_chol', 'A_smoothing','SA pre1/post0 (nonsym M)', 1, 0, 'chol',   2, 'sa',   [],  4, NaN) ...
    ];

    if smoke
        keep = ismember({configs.name}, ...
                        {'sa_pre1_post1_chol', 'sa_pre0_post0_chol', ...
                         'sa_pre1_post1_jac2', 'sa_pre1_post0_chol'});
        configs = configs(keep);
    end
end

function r = blank_row()
%BLANK_ROW  One row of results.csv, every field present so the struct array
%   concatenates and every writer sees a uniform schema.
    r = struct( ...
        'config', 'reference', 'section', 'reference', 'projector', '', ...
        'preSmooth', NaN, 'postSmooth', NaN, 'coarseSolve', '', ...
        'M_is_sym', true, 'coarseN', NaN, ...
        'k_target', NaN, 'rho', NaN, 'm_sketch', NaN, 'r_defl', NaN, 'q', NaN, ...
        'arm', '', 'sketch_only', true, ...
        'tau', NaN, 'tau_mode', 'sketch', 'tau_exact_ref', NaN, ...
        'tau_ritz_gap', NaN, ...
        'lam_min', NaN, 'lam_max', NaN, 'kappa', NaN, ...
        'kappa_ratio_vs_exact', NaN, 'lam_min_ratio', NaN, 'lam_max_ratio', NaN, ...
        'kappa_crosscheck', NaN, ...
        'iters', NaN, 'relres', NaN, 'flag', NaN, 'converged', false, ...
        'eigspace_err_2', NaN, 'angle_capture_frac_1pct', NaN, 'r_comp', NaN, ...
        'setup_sec_amg', NaN, 'setup_sec_V', NaN, 'spectrum_sec', NaN, ...
        'solve_sec', NaN, 'work_units_per_amg_apply', NaN, ...
        'ok', false, 'skip_reason', '');
end

function [r, sp] = run_arm(a, ctx, V, tau, base, A, b, tol, maxit, R, n, specOpts)
%RUN_ARM  Build one arm, measure its preconditioned spectrum, then solve.
%   Nothing here throws: a rank-deficient basis makes deflation_P_apply's chol
%   fail, and a nonsymmetric V-cycle invalidates both pcg and the symmetric
%   Lanczos.  Both outcomes are recorded as a skipped row rather than losing
%   the surrounding sweep.
    r        = base;
    r.arm    = a.id;
    r.sketch_only = a.sketch_only;
    sp       = struct('config', base.config, 'arm', a.id, ...
                      'k_target', base.k_target, 'q', base.q, ...
                      'ritz_low', [], 'ritz_high', []);

    if a.needs_sym && ~ctx.M_is_sym
        % preSmooth ~= postSmooth makes M nonsymmetric, so pcg and symmetric
        % Lanczos are both invalid.  Report the gap honestly instead of
        % printing numbers produced by an inapplicable method.
        r.skip_reason = 'M nonsymmetric (preSmooth ~= postSmooth)';
        return;
    end

    try
        s = a.build(ctx, V, tau);
    catch ME
        r.skip_reason = regexprep(ME.message, '\n.*', '');
        return;
    end
    if ~s.valid
        r.skip_reason = s.skip_reason;
        return;
    end

    t0   = tic;
    spec = precond_spectrum(s.prec, R, n, specOpts);
    r.spectrum_sec = toc(t0);
    if ~spec.ok
        r.skip_reason = spec.err;
        return;
    end
    r.lam_min   = spec.lam_min;
    r.lam_max   = spec.lam_max;
    r.kappa     = spec.kappa;
    sp.ritz_low  = spec.ritz_low;
    sp.ritz_high = spec.ritz_high;

    t1 = tic;
    try
        if isempty(s.prec)
            [~, fl, rr, it] = pcg(A, b, tol, maxit);
        else
            [~, fl, rr, it] = pcg(A, b, tol, maxit, s.prec);
        end
        r.iters = it;  r.relres = rr;  r.flag = fl;  r.converged = (fl == 0);
    catch ME
        r.skip_reason = regexprep(ME.message, '\n.*', '');
    end
    r.solve_sec = toc(t1);
    r.ok = true;
end

function r = finish_ratios(r, lam_min_ref, lam_max_ref, kappa_exact)
%FINISH_RATIOS  Normalize the two spectral ends against the unpreconditioned
%   spectrum, and kappa against the exact-eigenvector floor at the SAME
%   deflation dimension.  lam_max_ratio ~ 1 is the signature of deflation:
%   it never touches the top of the spectrum.
    if ~r.ok, return; end
    r.lam_min_ratio = r.lam_min / lam_min_ref;
    r.lam_max_ratio = r.lam_max / lam_max_ref;
    d = r.r_defl;
    if isfinite(d) && isKey(kappa_exact, d)
        r.kappa_ratio_vs_exact = r.kappa / kappa_exact(d);
    end
end

function s = kappa_summary(rws)
%KAPPA_SUMMARY  Compact per-arm kappa line for the progress log.
    parts = cell(1, numel(rws));
    for i = 1:numel(rws)
        if rws(i).ok
            parts{i} = sprintf('%s=%.3e', short_arm(rws(i).arm), rws(i).kappa);
        else
            parts{i} = sprintf('%s=skip', short_arm(rws(i).arm));
        end
    end
    s = strjoin(parts, ' ');
end

function s = short_arm(id)
    s = strrep(strrep(id, 'defl_amg_ichol', 'dfIC'), 'defl_amg', 'dfl');
    s = strrep(strrep(s, 'ctau_amg', 'ctau'), 'amg_direct', 'amg');
end

function [V, D] = load_or_compute_eigs_full(cacheDir, A, dA, k, h0)
%LOAD_OR_COMPUTE_EIGS_FULL  All k+1 cached smallest eigenpairs of A.
%   Same cache path and .mat format as load_or_compute_eigs_A in
%   run_amg_subspace_capture.m, so the two studies share one file; this variant
%   returns the FULL V and D because the reference floors are drawn at several
%   deflation dimensions, each needing its own lam_{d+1}.
    cachePath = fullfile(cacheDir, sprintf('eigsA_h%g_k%d.mat', h0, k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V = S.V;  D = S.D;
        fprintf('Loaded cached %s\n', cachePath);
        return;
    end
    fprintf('Computing smallest %d eigenpairs of A (one-time)...\n', k + 1);
    n  = size(A, 1);
    t0 = tic;
    % Name-value options, NOT the legacy opts struct: eigs silently ignores
    % struct fields with these capitalized names, losing symmetric mode.
    [Vraw, Dmat] = eigs(@(x) dA \ x, n, k + 1, 'smallestabs', ...
                        'Tolerance', 1e-10, 'MaxIterations', 5000, ...
                        'IsFunctionSymmetric', true);
    [D, idx] = sort(real(diag(Dmat)), 'ascend');
    V        = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
end

function write_defl_results_csv(csvPath, rows)
%WRITE_DEFL_RESULTS_CSV  One row per (config, k_target, q, arm).
    fid = fopen(csvPath, 'w');
    if fid < 0, error('Cannot open %s for writing.', csvPath); end
    c = onCleanup(@() fclose(fid));

    fprintf(fid, ['# Deflation basis is sketched from each config''s OWN AMG ', ...
                  'V-cycle, tau included: no exact spectral data enters any ', ...
                  'arm except the defl_exact references.\n']);
    fprintf(fid, ['# iters EXCLUDE the q*m_sketch V-cycle applies spent building ', ...
                  'the basis, and an AMG-preconditioned iteration costs about ', ...
                  'work_units_per_amg_apply x a deflation-only one.\n']);
    fprintf(fid, ['config,section,projector,preSmooth,postSmooth,coarseSolve,', ...
                  'M_is_sym,coarseN,k_target,rho,m_sketch,r_defl,q,arm,', ...
                  'sketch_only,tau,tau_mode,tau_exact_ref,tau_ritz_gap,', ...
                  'lam_min,lam_max,', ...
                  'kappa,kappa_ratio_vs_exact,lam_min_ratio,lam_max_ratio,', ...
                  'kappa_crosscheck,iters,relres,flag,converged,eigspace_err_2,', ...
                  'angle_capture_frac_1pct,r_comp,setup_sec_amg,setup_sec_V,', ...
                  'spectrum_sec,solve_sec,work_units_per_amg_apply,ok,', ...
                  'skip_reason\n']);
    for i = 1:numel(rows)
        r = rows(i);
        fprintf(fid, ['%s,%s,%s,%g,%g,%s,%d,%g,%g,%g,%g,%g,%g,%s,%d,', ...
                      '%.10g,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,', ...
                      '%.10g,%.10g,%g,%.6g,%g,%d,%.10g,%.10g,%g,%.6g,%.6g,', ...
                      '%.6g,%.6g,%.6g,%d,%s\n'], ...   % skip_reason pre-quoted
                r.config, r.section, r.projector, r.preSmooth, r.postSmooth, ...
                r.coarseSolve, r.M_is_sym, r.coarseN, r.k_target, r.rho, ...
                r.m_sketch, r.r_defl, r.q, r.arm, r.sketch_only, ...
                r.tau, r.tau_mode, r.tau_exact_ref, r.tau_ritz_gap, ...
                r.lam_min, r.lam_max, ...
                r.kappa, r.kappa_ratio_vs_exact, r.lam_min_ratio, r.lam_max_ratio, ...
                r.kappa_crosscheck, r.iters, r.relres, r.flag, r.converged, ...
                r.eigspace_err_2, r.angle_capture_frac_1pct, r.r_comp, ...
                r.setup_sec_amg, r.setup_sec_V, r.spectrum_sec, r.solve_sec, ...
                r.work_units_per_amg_apply, r.ok, csv_quote(r.skip_reason));
    end
end

function s = csv_quote(txt)
%CSV_QUOTE  RFC-4180 quote a free-text field.
%   skip_reason carries solver diagnostics that routinely contain commas
%   (e.g. "lam_min=-8e-16, lam_max=1.0"), which would otherwise shift every
%   column after it and silently corrupt the row.
    if isempty(txt)
        s = '""';
        return;
    end
    s = ['"', strrep(txt, '"', '""'), '"'];
end

function write_money_csv(csvPath, rows, configs, ktargets, qs, rho)
%WRITE_MONEY_CSV  The headline comparison, arms pivoted into columns.
%   One row per (config, k_target, q): subspace quality on the left, then the
%   condition number and both spectral ends for each arm side by side.
    armIds = {'amg_direct', 'defl_amg', 'defl_amg_ichol', 'ctau_amg'};

    fid = fopen(csvPath, 'w');
    if fid < 0, error('Cannot open %s for writing.', csvPath); end
    c = onCleanup(@() fclose(fid));

    hdr = 'config,k_target,m_sketch,r_defl,q,eigspace_err_2,tau_ritz_gap';
    for a = armIds
        hdr = [hdr, sprintf(',kappa_%s,lam_min_ratio_%s,lam_max_ratio_%s,iters_%s', ...
                            a{1}, a{1}, a{1}, a{1})];   %#ok<AGROW>
    end
    hdr = [hdr, ',kappa_defl_exact_ktarget,kappa_defl_exact_msketch,', ...
                'kappa_pcg_ichol,kappa_pcg_plain'];
    fprintf(fid, '%s\n', hdr);

    kap = @(id) pick(rows, strcmp({rows.arm}, id), 'kappa');
    kappa_ichol = kap('pcg_ichol');
    kappa_plain = kap('pcg_plain');

    for ic = 1:numel(configs)
        for kt = ktargets
            m_sk = rho * kt;
            for q = sort(qs, 'ascend')
                sel = strcmp({rows.config}, configs(ic).name) & ...
                      [rows.k_target] == kt & [rows.q] == q;
                selD = strcmp({rows.config}, configs(ic).name) & ...
                       strcmp({rows.arm}, 'amg_direct');
                if ~any(sel), continue; end
                sub = rows(sel);

                fprintf(fid, '%s,%g,%g,%g,%g,%.10g,%.10g', configs(ic).name, ...
                        kt, m_sk, sub(1).r_defl, q, sub(1).eigspace_err_2, ...
                        sub(1).tau_ritz_gap);
                for a = armIds
                    if strcmp(a{1}, 'amg_direct')
                        rsel = rows(selD);
                    else
                        rsel = sub(strcmp({sub.arm}, a{1}));
                    end
                    if isempty(rsel)
                        fprintf(fid, ',NaN,NaN,NaN,NaN');
                    else
                        fprintf(fid, ',%.10g,%.10g,%.10g,%g', rsel(1).kappa, ...
                                rsel(1).lam_min_ratio, rsel(1).lam_max_ratio, ...
                                rsel(1).iters);
                    end
                end
                fprintf(fid, ',%.10g,%.10g,%.10g,%.10g\n', ...
                        pick(rows, strcmp({rows.arm}, sprintf('defl_exact_d%d', kt)), 'kappa'), ...
                        pick(rows, strcmp({rows.arm}, sprintf('defl_exact_d%d', m_sk)), 'kappa'), ...
                        kappa_ichol, kappa_plain);
            end
        end
    end
end

function v = pick(rows, mask, field)
%PICK  First matching field value, NaN when the row is absent.
    idx = find(mask, 1);
    if isempty(idx)
        v = NaN;
    else
        v = rows(idx).(field);
    end
end
