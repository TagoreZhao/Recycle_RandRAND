% RUN_BENCHMARK  CCPP kernel-ridge PCG benchmark over a regularization sweep.
%
% Builds the dense RBF kernel K once on an n-point CCPP subset and, for each
% regularization value sigma2 in params.Sigma2List, solves the SPD system
%     A(sigma2) x = b,   A(sigma2) = K + sigma2*I,   b = standardized targets,
% with these PCG solvers:
%   unprec          : no preconditioner
%   ichol           : incomplete Cholesky (ict/droptol, via build_ichol_robust)
%   amg             : smoothed-aggregation AMG V-cycle with the ichol factor as
%                     fine smoother (make_amg_preconditioner)
%   defl_P          : deflation P = (I-VV') + tau*V(V'AV)^-1 V' applied directly
%                     on A, V = largest DEFLAT_LG_EIG eigenvectors of A
%   defl_sketch_q*  : same deflation P on A, but V from a Gaussian-sketched
%                     power iteration: Y = K^q * Omega with a shared start block
%                     Omega = randn(n, SKETCH_OVERSAMPLE*DEFLAT_LG_EIG), then one
%                     qr(Y,0); all p*k columns are kept (no Rayleigh-Ritz
%                     truncation).  One solver/curve per q in SKETCH_Q_LIST.
%   twolevel_VAhat  : two-level split scheme B = L^-T P L^-1 on Ahat = L^-1 A L^-T
%                     with V = largest eigenvectors of Ahat
% The two-level application mirrors Preconditioner_Recycle/report/
% ball_surface_krylov_recycle/solve_krylov_recycle_surface.m:
%   Papply = deflation_P_apply(V, Ahat_fun, tau);  Bapply = @(r) Lt\(Papply(L\r)).
%
% Recycling: params.DEFLAT_PREC_REFRESH rebuilds the deflation bases every N
% sigma2 values (Inf = build once at the first sigma2 and recycle).  V_A
% recycling is EXACT (A = K + sigma2*I shares K's eigenvectors); V_T recycling
% is approximate (the ichol factor L changes with sigma2).
%
% Output (benchmark_sigma2/): all_results.csv, iterations_vs_sigma2.png,
% run_config.{mat,json}.  The plot shows converged runs only (flag == 0);
% non-converged solves appear as gaps (the CSV keeps their raw iter/flag/
% relres -- note pcg returns the *best* iterate's index on failure, not a
% meaningful iteration count).  Set SMOKE_TEST=1 in the base workspace for a
% fast small-n end-to-end check (writes to benchmark_sigma2_smoke/).

%% ===================== 1. Setup / params ==================================
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);
addpath(thisFileDir);

params.n          = 3000;                 % kernel subset size
params.Sigma2List = logspace(-8, 0, 10);  % regularization sweep (x axis)
params.EllFactor  = 1.0;                  % ell = EllFactor * median heuristic
params.Tol        = 1e-6;   % eps*cond(A) ~ 1e-5 at sigma2=1e-8; 1e-8 is unattainable there
params.MaxIt      = 10000;
params.Seed       = 0;

params.DEFLAT_LG_EIG       = 100;   % # largest eigenvectors in the coarse space
params.DEFLAT_SM_EIG       = 0;     % recorded only; largest-only deflation here
params.DEFLAT_TAU          = 0.5;   % coarse-correction weight tau
params.DEFLAT_PREC_REFRESH = Inf;   % rebuild V every N sigma2; Inf = recycle

params.SKETCH_Q_LIST     = [1 2 3]; % # K-applies per sketched basis (one curve each)
params.SKETCH_OVERSAMPLE = 2;       % multiplicative: sketch width = p * DEFLAT_LG_EIG

params.icholOpts = struct('type', 'ict', 'droptol', 1e-3, 'michol', 'on');
params.amgOpts   = struct('maxLevels', 2, 'minCoarseSize', 800, ...
                          'theta', 0.05, 'omegaInterp', 0);

outName = 'benchmark_sigma2';
if evalin('base', 'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)')
    fprintf('[SMOKE_TEST] Overriding params for fast end-to-end check.\n');
    params.n = 400;
    params.Sigma2List = logspace(-8, 0, 3);
    params.DEFLAT_LG_EIG = 20;
    params.MaxIt = 500;
    outName = 'benchmark_sigma2_smoke';
end
outDir = fullfile(thisFileDir, outName);
if ~exist(outDir, 'dir'), mkdir(outDir); end

solver_keys   = {'unprec', 'ichol', 'amg', 'defl_P', 'twolevel_VAhat'};
solver_labels = {'PCG (no prec)', 'PCG + ichol', 'PCG + AMG (ichol smoother)', ...
                 'PCG + deflation P on A (V from A)', ...
                 'PCG + two-level split (V from Ahat)'};
sketch_keys   = arrayfun(@(q) sprintf('defl_sketch_q%d', q), ...
                         params.SKETCH_Q_LIST, 'UniformOutput', false);
sketch_labels = arrayfun(@(q) sprintf('PCG + deflation P on A (sketch q=%d, p=%d)', ...
                         q, params.SKETCH_OVERSAMPLE), ...
                         params.SKETCH_Q_LIST, 'UniformOutput', false);
solver_keys   = [solver_keys,   sketch_keys];
solver_labels = [solver_labels, sketch_labels];

%% ===================== 2. Data + kernel (fixed for the sweep) =============
rng(params.Seed);
[X, y]  = load_dataset_csv_or_mat(fullfile(thisFileDir, 'data', 'ccpp.csv'));
[Xs, ys] = standardize_data(X, y);
perm = randperm(size(Xs, 1));
idx  = perm(1:params.n);
Xn   = Xs(idx, :);
b    = ys(idx);

ell0 = estimate_median_lengthscale(Xn);
ell  = params.EllFactor * ell0;
fprintf('[ccpp] n=%d  ell0=%.4f  ell=%.4f  building dense K ...\n', params.n, ell0, ell);
K    = rbf_kernel_matrix(Xn, ell);
K_sp = sparse(K);                  % one conversion, reused for ichol/AMG builds

n   = params.n;
tau = params.DEFLAT_TAU;
kLg = params.DEFLAT_LG_EIG;
nS  = numel(params.Sigma2List);

%% ===================== 3. Sweep over Sigma2List ===========================
R = struct();
for s = 1:numel(solver_keys)
    R.([solver_keys{s} '_its'])    = nan(nS, 1);
    R.([solver_keys{s} '_flag'])   = -ones(nS, 1);
    R.([solver_keys{s} '_relres']) = nan(nS, 1);
    R.([solver_keys{s} '_time'])   = nan(nS, 1);
end
R.ichol_build_time = nan(nS, 1);  R.ichol_alpha    = nan(nS, 1);
R.amg_build_time   = nan(nS, 1);
R.VA_build_time    = nan(nS, 1);  R.VT_build_time  = nan(nS, 1);
R.defl_rebuilt     = zeros(nS, 1);
for q = params.SKETCH_Q_LIST
    R.(sprintf('sketch_q%d_build_time', q)) = nan(nS, 1);
end

V_A  = [];
V_T  = [];
V_sk = struct();
for j = 1:nS
    sigma2 = params.Sigma2List(j);
    fprintf('\n===== sigma2 %d/%d = %.3e =====\n', j, nS, sigma2);
    Amul = @(x) K*x + sigma2*x;          % dense matvec; never form dense A
    A_sp = K_sp + sigma2*speye(n);

    % --- ichol (rebuilt every sigma2) ---
    icholOK = true;
    try
        [L, alphaUsed, tIchol] = build_ichol_robust(A_sp, params.icholOpts);
        Lt = L';
        R.ichol_alpha(j) = alphaUsed;
        R.ichol_build_time(j) = tIchol;
    catch ME
        warning('run_benchmark:icholFailed', ...
                'ichol build failed at sigma2=%.3e: %s', sigma2, ME.message);
        icholOK = false;
    end

    % --- AMG with ichol fine smoother (rebuilt every sigma2) ---
    amgOK = false;
    if icholOK
        try
            tA = tic;
            Mamg = src.precond.make_amg_preconditioner(A_sp, ...
                'maxLevels',     params.amgOpts.maxLevels, ...
                'minCoarseSize', min(params.amgOpts.minCoarseSize, floor(n/4)), ...
                'theta',         params.amgOpts.theta, ...
                'omegaInterp',   params.amgOpts.omegaInterp, ...
                'fineSmootherL',  L, ...
                'fineSmootherLt', Lt);
            R.amg_build_time(j) = toc(tA);
            amgOK = true;
        catch ME
            warning('run_benchmark:amgFailed', ...
                    'AMG build failed at sigma2=%.3e: %s', sigma2, ME.message);
        end
    end
    if icholOK
        Ahat_fun = @(x) L \ (Amul(Lt \ x));   % split operator L^-1 A L^-T
    end

    % --- deflation-basis refresh cadence ---
    rebuildV = (j == 1) || (isfinite(params.DEFLAT_PREC_REFRESH) && ...
                            mod(j-1, params.DEFLAT_PREC_REFRESH) == 0);
    if rebuildV
        R.defl_rebuilt(j) = 1;
        fprintf('  rebuilding deflation bases (%d largest eigenvectors)\n', kLg);
        % V_A from K: A = K + sigma2*I shares K's eigenvectors, so recycling
        % V_A across the sweep is exact (this rebuild is a mathematical no-op).
        tV = tic;
        [V_A, ~] = eigs(K, kLg, 'largestabs', 'Tolerance', 1e-8);
        [V_A, ~] = qr(V_A, 0);
        R.VA_build_time(j) = toc(tV);
        % Sketched bases: shared Gaussian start block, one basis per q.  Like
        % V_A these come from K alone, so recycling across sigma2 is exact.
        Omega = randn(n, params.SKETCH_OVERSAMPLE * kLg);
        for q = params.SKETCH_Q_LIST
            tV = tic;
            Y = src.precond.subspace_iter_plain(@(X) K*X, Omega, q);  % K^q * Omega
            [Vq, ~] = qr(Y, 0);
            V_sk.(sprintf('defl_sketch_q%d', q)) = Vq;
            R.(sprintf('sketch_q%d_build_time', q))(j) = toc(tV);
        end
        if icholOK
            tV = tic;
            [V_T, ~] = eigs(Ahat_fun, n, kLg, 'largestabs', ...
                            'IsFunctionSymmetric', true, 'Tolerance', 1e-6);
            [V_T, ~] = qr(V_T, 0);
            R.VT_build_time(j) = toc(tV);
        end
    end

    % --- solves ---
    R = run_solver(R, j, 'unprec', @() pcg(Amul, b, params.Tol, params.MaxIt));
    if icholOK
        R = run_solver(R, j, 'ichol', @() pcg(Amul, b, params.Tol, params.MaxIt, L, Lt));
    end
    if amgOK
        R = run_solver(R, j, 'amg', @() pcg(Amul, b, params.Tol, params.MaxIt, Mamg));
    end
    R = run_solver(R, j, 'defl_P', @() solve_defl_on_A(Amul, b, params, V_A, tau));
    for q = params.SKETCH_Q_LIST
        key = sprintf('defl_sketch_q%d', q);
        R = run_solver(R, j, key, @() solve_defl_on_A(Amul, b, params, V_sk.(key), tau));
    end
    if icholOK && ~isempty(V_T)
        R = run_solver(R, j, 'twolevel_VAhat', ...
            @() solve_two_level(Amul, Ahat_fun, b, params, V_T, tau, L, Lt));
    end
end

%% ===================== 4. CSV + plot + config =============================
T = table((1:nS)', params.Sigma2List(:), repmat(n, nS, 1), ...
          repmat(ell, nS, 1), repmat(ell0, nS, 1), ...
          'VariableNames', {'j', 'sigma2', 'n', 'ell', 'ell0'});
for s = 1:numel(solver_keys)
    key = solver_keys{s};
    T.([key '_its'])    = R.([key '_its']);
    T.([key '_flag'])   = R.([key '_flag']);
    T.([key '_relres']) = R.([key '_relres']);
    T.([key '_time'])   = R.([key '_time']);
end
T.ichol_build_time = R.ichol_build_time;  T.ichol_alpha   = R.ichol_alpha;
T.amg_build_time   = R.amg_build_time;
T.VA_build_time    = R.VA_build_time;     T.VT_build_time = R.VT_build_time;
for q = params.SKETCH_Q_LIST
    T.(sprintf('sketch_q%d_build_time', q)) = R.(sprintf('sketch_q%d_build_time', q));
end
T.defl_rebuilt     = R.defl_rebuilt;
writetable(T, fullfile(outDir, 'all_results.csv'));
fprintf('\nWrote %s\n', fullfile(outDir, 'all_results.csv'));

fh = figure('Visible', 'off', 'Position', [100 100 760 440]);
markers = {'-o', '-s', '-^', '-d', '-v', '-p', '-h', '-x', '-*', '-+'};
hold on;
for s = 1:numel(solver_keys)
    its  = max(R.([solver_keys{s} '_its']), 1);   % clamp before NaN: max(NaN,1)==1
    flag = R.([solver_keys{s} '_flag']);
    its(flag ~= 0) = NaN;              % non-converged -> gap in the line
    loglog(params.Sigma2List(:), its, markers{mod(s-1, numel(markers)) + 1}, ...
           'MarkerSize', 4, 'LineWidth', 1.2);
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlim([min(params.Sigma2List), max(params.Sigma2List)]);  % keep failed-run gaps visible
grid on; xlabel('\sigma^2'); ylabel('PCG iterations');
legend(solver_labels, 'Location', 'best', 'Interpreter', 'none');
title(sprintf('CCPP kernel PCG: iterations vs regularization (n=%d, %d largest eigvecs)', ...
              n, kLg), 'Interpreter', 'none');
saveas(fh, fullfile(outDir, 'iterations_vs_sigma2.png')); close(fh);
fprintf('Wrote %s\n', fullfile(outDir, 'iterations_vs_sigma2.png'));

cfg_out = struct();
cfg_out.params        = params;
cfg_out.solver_keys   = {solver_keys{:}};                       %#ok<CCAT1>
cfg_out.solver_labels = {solver_labels{:}};                     %#ok<CCAT1>
cfg_out.data_file     = fullfile(thisFileDir, 'data', 'ccpp.csv');
cfg_out.ell0          = ell0;
cfg_out.ell           = ell;
cfg_out.smoke_test    = strcmp(outName, 'benchmark_sigma2_smoke');
cfg_out.notes = {'AMG reuses the ichol factor as fine smoother.', ...
                 'V_A recycling is exact (eigenvectors of K); V_T recycling is approximate (L changes with sigma2).', ...
                 'Sketched bases: Y = K^q * Omega (subspace_iter_plain), Omega = randn(n, p*k) shared across q, all p*k columns kept after qr(Y,0); K-based so recycling is exact.', ...
                 'Two-level = split scheme B = L^-T P L^-1 on Ahat = L^-1 A L^-T (ball_surface_krylov_recycle convention).'};
save(fullfile(outDir, 'run_config.mat'), 'cfg_out');
fid = fopen(fullfile(outDir, 'run_config.json'), 'w');
if fid > 0, fwrite(fid, jsonencode(cfg_out)); fclose(fid); end

fprintf('\n[ccpp run_benchmark] done. Output in %s\n', outDir);

%==========================================================================
%  Local functions
%==========================================================================
function R = run_solver(R, j, key, solveFun)
%RUN_SOLVER  Time one build+solve closure; record its/flag/relres, never abort.
    t0 = tic;
    try
        [~, flag, relres, iter] = solveFun();
    catch ME
        warning('run_benchmark:solverFailed', '%s failed at j=%d: %s', ...
                key, j, ME.message);
        flag = -1; relres = NaN; iter = NaN;
    end
    R.([key '_time'])(j)   = toc(t0);
    R.([key '_its'])(j)    = iter;
    R.([key '_flag'])(j)   = flag;
    R.([key '_relres'])(j) = relres;
    fprintf('  %-15s its=%4g  flag=%2d  relres=%.2e\n', key, iter, flag, relres);
end

function [x, flag, relres, iter] = solve_defl_on_A(Amul, b, params, V, tau)
%SOLVE_DEFL_ON_A  Pure deflation preconditioner P applied directly on A.
    Papply = src.precond.deflation_P_apply(V, Amul, tau);
    [x, flag, relres, iter] = pcg(Amul, b, params.Tol, params.MaxIt, @(r) Papply(r));
end

function [x, flag, relres, iter] = solve_two_level(Amul, Ahat_fun, b, params, V, tau, L, Lt)
%SOLVE_TWO_LEVEL  Split two-level scheme B = L^-T P L^-1 on Ahat = L^-1 A L^-T.
    Papply = src.precond.deflation_P_apply(V, Ahat_fun, tau);
    Bapply = @(r) Lt \ (Papply(L \ r));
    [x, flag, relres, iter] = pcg(Amul, b, params.Tol, params.MaxIt, @(r) Bapply(r));
end
