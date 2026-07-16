% RUN_BENCHMARK  Duke logistic-regression Newton-PCG preconditioner benchmark.
%
% Runs an instrumented Newton loop for L2-penalized logistic regression on the
% duke dataset (n=38 samples, d=7129 features) and, at every Newton iteration,
% solves the identical Newton system
%     H_k delta = -g_k,   H_k = Xa'diag(w_k)Xa + lambda*I,   Xa = [X, 1]
% with the same five PCG solvers as GP_train/ccpp/run_benchmark.m:
%   unprec         : no preconditioner
%   ichol          : incomplete Cholesky (ict/droptol, via build_ichol_robust)
%   amg            : smoothed-aggregation AMG V-cycle with the ichol factor as
%                    fine smoother (make_amg_preconditioner)
%   defl_P         : deflation P = (I-VV') + tau*V(V'HV)^-1 V' applied directly
%                    on H, V = largest DEFLAT_LG_EIG eigenvectors of H
%   twolevel_VAhat : two-level split scheme B = L^-T P L^-1 on Ahat = L^-1 H L^-T
%                    with V = largest eigenvectors of Ahat
%
% Fairness: all five solvers see the identical (H_k, -g_k) because the Newton
% trajectory is advanced only by an exact Woodbury reference solve
%     H^-1 r = (r - B'((lambda*I_n + BB')\(B r)))/lambda,   B = sqrt(w).*Xa
% (rank-n data term, 38x38 inner solve), followed by the same descent-guard +
% Armijo line search as logreg_newton. The benchmarked solutions are discarded.
%
% Sweep: outer loop over params.LambdaList; one figure per lambda with x-axis =
% Newton iteration, y-axis = PCG iterations (log scale).
%
% Recycling: params.DEFLAT_PREC_REFRESH rebuilds the deflation bases every N
% Newton iterations within a lambda (Inf = build at Newton iter 1 and recycle);
% bases are rebuilt fresh at the start of each lambda. Unlike ccpp (where V_A
% recycling is exact), recycling here is approximate for BOTH bases: H_k
% changes with the IRLS weights every step and the ichol factor L is rebuilt
% every step. deflation_P_apply recomputes E = V'HV with the current operator
% on every solve, so recycled V degrades preconditioner quality gracefully --
% that degradation across the Newton path is the phenomenon being measured.
%
% Caveat: ichol/AMG need an explicit H; gene-expression X is dense-ish so the
% sparse H_k is near-dense (~m^2 nnz, m = d+1 = 7130). nnz is printed once per
% lambda. unprec/defl_P stay matrix-free via hessian_operator.
%
% Output (benchmark_newton/): all_results.csv (long format, one row per
% (lambda, newton_iter)), iterations_vs_newton_lambda_<val>.png per lambda,
% run_config.{mat,json}. Plots show converged runs only (flag == 0);
% non-converged solves appear as gaps (the CSV keeps their raw iter/flag/
% relres -- pcg returns the *best* iterate's index on failure, not a
% meaningful iteration count). Set SMOKE_TEST=1 in the base workspace for a
% fast end-to-end check (subsampled features, writes to benchmark_newton_smoke/).

%% ===================== 1. Setup / params ==================================
thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(fileparts(thisFileDir));
addpath(repoRoot);                                % +src package
addpath(fullfile(repoRoot, 'GP_train', 'ccpp'));  % build_ichol_robust
addpath(thisFileDir);                             % last: duke helpers win

params.LambdaList    = logspace(-4, 2, 4);  % regularization sweep (one plot each)
params.NewtonMaxIter = 50;
params.NewtonTol     = 1e-8;    % relative grad-norm (logreg_newton default)
params.Tol           = 1e-6;    % PCG tolerance
params.MaxIt         = 2000;    % PCG max iterations
params.Seed          = 0;

params.DEFLAT_LG_EIG       = 40;   % data term has rank <= n=38; ~40 captures
                                   % everything above the lambda floor
params.DEFLAT_TAU          = 0.5;  % coarse-correction weight tau
params.DEFLAT_PREC_REFRESH = Inf;  % rebuild V every N Newton iters; Inf = recycle

params.icholOpts = struct('type', 'ict', 'droptol', 1e-3, 'michol', 'on');
% Near-dense, near-rank-n H needs much larger diagcomp shifts than the
% build_ichol_robust defaults before ict finds positive pivots.
params.icholAlphaList = [0 0.01 0.1 0.5 1 5 25 100 1e3 1e4];
params.amgOpts   = struct('maxLevels', 2, 'minCoarseSize', 800, ...
                          'theta', 0.05, 'omegaInterp', 0);

outName   = 'benchmark_newton';
smokeCols = 0;                     % 0 = keep all feature columns
if evalin('base', 'exist(''SMOKE_TEST'',''var'') && logical(SMOKE_TEST)')
    fprintf('[SMOKE_TEST] Overriding params for fast end-to-end check.\n');
    params.LambdaList    = logspace(-4, 0, 2);
    params.NewtonMaxIter = 5;
    params.DEFLAT_LG_EIG = 10;
    params.MaxIt         = 500;
    smokeCols = 800;               % subsample features -> small explicit H
    outName   = 'benchmark_newton_smoke';
end
outDir = fullfile(thisFileDir, outName);
if ~exist(outDir, 'dir'), mkdir(outDir); end

solver_keys   = {'unprec', 'ichol', 'amg', 'defl_P', 'twolevel_VAhat'};
solver_labels = {'PCG (no prec)', 'PCG + ichol', 'PCG + AMG (ichol smoother)', ...
                 'PCG + deflation P on H (V from H)', ...
                 'PCG + two-level split (V from Ahat)'};

%% ===================== 2. Data (fixed for the sweep) ======================
rng(params.Seed);
dataFile = fullfile(thisFileDir, 'data', 'duke.tr');
if ~exist(dataFile, 'file')
    error('run_benchmark:noData', ...
          'Missing %s -- run download_duke() first.', dataFile);
end
[Xraw, y] = load_libsvm(dataFile);
Xs = standardize_features(Xraw);
if smokeCols > 0
    Xs = Xs(:, 1:min(smokeCols, size(Xs, 2)));
end
[n, dEff] = size(Xs);
m     = dEff + 1;                       % system size: [weights; bias]
Xa_sp = [Xs, sparse(ones(n, 1))];       % n-by-m, for explicit H and Woodbury
fprintf('[duke] n=%d  d=%d  system size m=%d\n', n, dEff, m);

tau = params.DEFLAT_TAU;
kLg = min(params.DEFLAT_LG_EIG, m - 2);
nL  = numel(params.LambdaList);

%% ===================== 3. Results container (long format) =================
maxRows = nL * params.NewtonMaxIter;
R = struct();
for s = 1:numel(solver_keys)
    R.([solver_keys{s} '_its'])    = nan(maxRows, 1);
    R.([solver_keys{s} '_flag'])   = -ones(maxRows, 1);
    R.([solver_keys{s} '_relres']) = nan(maxRows, 1);
    R.([solver_keys{s} '_time'])   = nan(maxRows, 1);
end
R.lambda_idx  = nan(maxRows, 1);  R.lambda      = nan(maxRows, 1);
R.newton_iter = nan(maxRows, 1);
R.gnorm       = nan(maxRows, 1);  R.relgnorm    = nan(maxRows, 1);
R.obj         = nan(maxRows, 1);  R.step_t      = nan(maxRows, 1);
R.ref_solve_time   = nan(maxRows, 1);
R.ichol_build_time = nan(maxRows, 1);  R.ichol_alpha   = nan(maxRows, 1);
R.amg_build_time   = nan(maxRows, 1);
R.VH_build_time    = nan(maxRows, 1);  R.VT_build_time = nan(maxRows, 1);
R.defl_rebuilt     = zeros(maxRows, 1);

lam_newton_iters = zeros(nL, 1);
lam_converged    = false(nL, 1);
lam_train_acc    = nan(nL, 1);
lam_separable    = false(nL, 1);

markers  = {'-o', '-s', '-^', '-d', '-v'};
rowCount = 0;

%% ===================== 4. Sweep over LambdaList ===========================
for j = 1:nL
    lambda = params.LambdaList(j);
    fprintf('\n===== lambda %d/%d = %.3e =====\n', j, nL, lambda);

    beta = zeros(m, 1);
    [f, g, p] = logreg_obj(beta, Xs, y, lambda);
    g0  = max(norm(g), eps);
    V_H = [];                            % fresh deflation bases per lambda
    V_T = [];
    printedNnz = false;

    it = 0;
    while it < params.NewtonMaxIter
        if norm(g) <= params.NewtonTol * g0
            lam_converged(j) = true;
            break;
        end
        it = it + 1;
        rowCount = rowCount + 1;
        r = rowCount;
        R.lambda_idx(r)  = j;
        R.lambda(r)      = lambda;
        R.newton_iter(r) = it;
        fprintf('--- newton iter %d (lambda=%.3e) ---\n', it, lambda);

        % --- operators for this Newton system ---
        w    = p .* (1 - p);
        Hmul = hessian_operator(Xs, w, lambda);   % matrix-free H*v
        rhs  = -g;
        B_sp = spdiags(sqrt(w), 0, n, n) * Xa_sp; % n-by-m
        H_sp = B_sp' * B_sp + lambda * speye(m);  % explicit H for ichol/AMG
        if ~printedNnz
            fprintf('  H is %dx%d, nnz=%d (%.1f%% dense)\n', ...
                    m, m, nnz(H_sp), 100 * nnz(H_sp) / m^2);
            printedNnz = true;
        end

        % --- ichol (rebuilt every Newton step) ---
        icholOK = true;
        try
            [L, alphaUsed, tIchol] = build_ichol_robust(H_sp, params.icholOpts, ...
                                                        params.icholAlphaList);
            Lt = L';
            R.ichol_alpha(r) = alphaUsed;
            R.ichol_build_time(r) = tIchol;
        catch ME
            warning('run_benchmark:icholFailed', ...
                    'ichol build failed at lambda=%.3e it=%d: %s', ...
                    lambda, it, ME.message);
            icholOK = false;
        end

        % --- AMG with ichol fine smoother (rebuilt every Newton step) ---
        amgOK = false;
        if icholOK
            try
                tA = tic;
                Mamg = src.precond.make_amg_preconditioner(H_sp, ...
                    'maxLevels',     params.amgOpts.maxLevels, ...
                    'minCoarseSize', min(params.amgOpts.minCoarseSize, floor(m/4)), ...
                    'theta',         params.amgOpts.theta, ...
                    'omegaInterp',   params.amgOpts.omegaInterp, ...
                    'fineSmootherL',  L, ...
                    'fineSmootherLt', Lt);
                R.amg_build_time(r) = toc(tA);
                amgOK = true;
            catch ME
                warning('run_benchmark:amgFailed', ...
                        'AMG build failed at lambda=%.3e it=%d: %s', ...
                        lambda, it, ME.message);
            end
        end
        if icholOK
            Ahat_fun = @(x) L \ (Hmul(Lt \ x));   % split operator L^-1 H L^-T
        end

        % --- deflation-basis refresh cadence (within this lambda) ---
        rebuildV = (it == 1) || (isfinite(params.DEFLAT_PREC_REFRESH) && ...
                                 mod(it-1, params.DEFLAT_PREC_REFRESH) == 0);
        if rebuildV
            R.defl_rebuilt(r) = 1;
            fprintf('  rebuilding deflation bases (%d largest eigenvectors)\n', kLg);
            try
                tV = tic;
                [Vnew, ~] = eigs(Hmul, m, kLg, 'largestabs', ...
                                 'IsFunctionSymmetric', true, 'Tolerance', 1e-6);
                [V_H, ~] = qr(Vnew, 0);
                R.VH_build_time(r) = toc(tV);
            catch ME
                warning('run_benchmark:VHFailed', ...
                        'eigs for V_H failed at lambda=%.3e it=%d: %s (keeping previous V_H)', ...
                        lambda, it, ME.message);
            end
        end
        % V_T also rebuilds on the first ichol-OK step after a failed rebuild
        % (an it==1 ichol failure would otherwise leave V_T empty for good).
        if icholOK && (rebuildV || isempty(V_T))
            try
                tV = tic;
                [Vnew, ~] = eigs(Ahat_fun, m, kLg, 'largestabs', ...
                                 'IsFunctionSymmetric', true, 'Tolerance', 1e-6);
                [V_T, ~] = qr(Vnew, 0);
                R.VT_build_time(r) = toc(tV);
            catch ME
                warning('run_benchmark:VTFailed', ...
                        'eigs for V_T failed at lambda=%.3e it=%d: %s (keeping previous V_T)', ...
                        lambda, it, ME.message);
            end
        end

        % --- benchmarked solves (identical system; solutions discarded) ---
        R = run_solver(R, r, 'unprec', @() pcg(Hmul, rhs, params.Tol, params.MaxIt));
        if icholOK
            R = run_solver(R, r, 'ichol', @() pcg(Hmul, rhs, params.Tol, params.MaxIt, L, Lt));
        end
        if amgOK
            R = run_solver(R, r, 'amg', @() pcg(Hmul, rhs, params.Tol, params.MaxIt, Mamg));
        end
        if ~isempty(V_H)
            R = run_solver(R, r, 'defl_P', @() solve_defl_on_A(Hmul, rhs, params, V_H, tau));
        end
        if icholOK && ~isempty(V_T)
            R = run_solver(R, r, 'twolevel_VAhat', ...
                @() solve_two_level(Hmul, Ahat_fun, rhs, params, V_T, tau, L, Lt));
        end

        % --- exact Woodbury reference step (advances the trajectory) ---
        tRef  = tic;
        C     = full(B_sp * B_sp') + lambda * eye(n);          % n-by-n SPD
        delta = (rhs - B_sp' * (C \ (B_sp * rhs))) / lambda;
        R.ref_solve_time(r) = toc(tRef);

        % --- descent guard + Armijo backtracking (as in logreg_newton) ---
        gd = g' * delta;
        if ~(gd < 0)
            delta = -g;
            gd = g' * delta;
        end
        t = 1; c1 = 1e-4;
        fnew = logreg_obj(beta + t * delta, Xs, y, lambda);
        while fnew > f + c1 * t * gd && t > 1e-10
            t = 0.5 * t;
            fnew = logreg_obj(beta + t * delta, Xs, y, lambda);
        end
        beta = beta + t * delta;
        [f, g, p] = logreg_obj(beta, Xs, y, lambda);
        R.step_t(r)   = t;
        R.obj(r)      = f;
        R.gnorm(r)    = norm(g);
        R.relgnorm(r) = norm(g) / g0;
        clear B_sp H_sp
    end

    lam_newton_iters(j) = it;
    lam_train_acc(j)    = mean(double(p >= 0.5) == y);
    lam_separable(j)    = lam_train_acc(j) >= 1 - eps;
    fprintf('lambda=%.3e: newton_iters=%d converged=%d train_acc=%.3f\n', ...
            lambda, it, lam_converged(j), lam_train_acc(j));

    % --- per-lambda plot: PCG iterations vs Newton iteration ---
    rows = find(R.lambda_idx(1:rowCount) == j);
    if isempty(rows), continue; end
    x  = R.newton_iter(rows);
    fh = figure('Visible', 'off', 'Position', [100 100 760 440]);
    hold on;
    for s = 1:numel(solver_keys)
        its  = max(R.([solver_keys{s} '_its'])(rows), 1);  % clamp before NaN
        flag = R.([solver_keys{s} '_flag'])(rows);
        its(flag ~= 0) = NaN;           % non-converged -> gap in the line
        semilogy(x, its, markers{s}, 'MarkerSize', 4, 'LineWidth', 1.2);
    end
    set(gca, 'YScale', 'log');
    xlim([1, max(max(x), 2)]);
    xticks(unique(round(linspace(1, max(x), min(max(x), 10)))));
    grid on; xlabel('Newton iteration'); ylabel('PCG iterations');
    legend(solver_labels, 'Location', 'best', 'Interpreter', 'none');
    title(sprintf('duke logistic Newton PCG: lambda=%.0e (n=%d, d=%d, %d largest eigvecs)', ...
                  lambda, n, dEff, kLg), 'Interpreter', 'none');
    tok = strrep(sprintf('%.0e', lambda), '+', '');
    saveas(fh, fullfile(outDir, sprintf('iterations_vs_newton_lambda_%s.png', tok)));
    close(fh);
    fprintf('Wrote %s\n', fullfile(outDir, sprintf('iterations_vs_newton_lambda_%s.png', tok)));
end

%% ===================== 5. CSV + config ====================================
fn = fieldnames(R);
for s = 1:numel(fn)
    R.(fn{s}) = R.(fn{s})(1:rowCount);
end

T = table(R.lambda_idx, R.lambda, R.newton_iter, R.gnorm, R.relgnorm, ...
          R.obj, R.step_t, R.ref_solve_time, ...
          'VariableNames', {'lambda_idx', 'lambda', 'newton_iter', 'gnorm', ...
                            'relgnorm', 'obj', 'step_t', 'ref_solve_time'});
for s = 1:numel(solver_keys)
    key = solver_keys{s};
    T.([key '_its'])    = R.([key '_its']);
    T.([key '_flag'])   = R.([key '_flag']);
    T.([key '_relres']) = R.([key '_relres']);
    T.([key '_time'])   = R.([key '_time']);
end
T.ichol_build_time = R.ichol_build_time;  T.ichol_alpha   = R.ichol_alpha;
T.amg_build_time   = R.amg_build_time;
T.VH_build_time    = R.VH_build_time;     T.VT_build_time = R.VT_build_time;
T.defl_rebuilt     = R.defl_rebuilt;
writetable(T, fullfile(outDir, 'all_results.csv'));
fprintf('\nWrote %s\n', fullfile(outDir, 'all_results.csv'));

cfg_out = struct();
cfg_out.params        = params;
cfg_out.solver_keys   = {solver_keys{:}};                       %#ok<CCAT1>
cfg_out.solver_labels = {solver_labels{:}};                     %#ok<CCAT1>
cfg_out.data_file     = dataFile;
cfg_out.n             = n;
cfg_out.d             = dEff;
cfg_out.m             = m;
cfg_out.newton_iters  = lam_newton_iters;
cfg_out.converged     = lam_converged;
cfg_out.train_acc     = lam_train_acc;
cfg_out.separable     = lam_separable;
cfg_out.smoke_test    = strcmp(outName, 'benchmark_newton_smoke');
cfg_out.notes = {'All five solvers solve the identical (H_k, -g_k) per Newton step; trajectory advanced only by the exact Woodbury reference solve.', ...
                 'V_H and V_T recycling are both approximate across Newton steps (H_k and L change); E = V''HV is recomputed with the current operator each solve.', ...
                 'H_sp is near-dense (rank-n data term on dense-ish gene-expression X) -- ichol/AMG build cost caveat.', ...
                 'Two-level = split scheme B = L^-T P L^-1 on Ahat = L^-1 H L^-T (ball_surface_krylov_recycle convention).'};
save(fullfile(outDir, 'run_config.mat'), 'cfg_out');
fid = fopen(fullfile(outDir, 'run_config.json'), 'w');
if fid > 0, fwrite(fid, jsonencode(cfg_out)); fclose(fid); end

fprintf('\n[duke run_benchmark] done. Output in %s\n', outDir);

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

function [f, g, p] = logreg_obj(beta, X, y, lambda)
%LOGREG_OBJ  Penalized logistic objective, gradient, and probabilities.
    d   = numel(beta) - 1;
    wt  = beta(1:d);
    b   = beta(end);
    eta = X * wt + b;
    p   = sigmoid(eta);

    % Stable negative log-likelihood: softplus(eta) - y.*eta.
    ll = softplus(eta) - y .* eta;
    f  = sum(ll) + 0.5 * lambda * (beta' * beta);

    if nargout > 1
        r = p - y;                              % n-by-1 residual
        g = [X' * r + lambda * wt; sum(r) + lambda * b];
    end
end

function s = softplus(z)
%SOFTPLUS  Numerically stable log(1 + exp(z)).
    s = max(z, 0) + log1p(exp(-abs(z)));
end
