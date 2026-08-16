function outputs = run_ard_training_benchmark(varargin)
%RUN_ARD_TRAINING_BENCHMARK  End-to-end ARD GP training with recycled PCG bases.
%   OUTPUTS = RUN_ARD_TRAINING_BENCHMARK() trains independent exact-GP arms on
%   CCPP.  Each Adam state changes four ARD lengthscales, signal variance, and
%   noise variance.  Marginal-likelihood gradients use PCG solves for the
%   target and fixed Hutchinson probes.
%
%   Name-value options include:
%     'SmokeTest'     logical, use the fast n=300 configuration (false)
%     'DataFile'      CCPP CSV path
%     'OutDir'        output directory
%     'NTrain'        number of training examples (3000)
%     'NTest'         held-out examples (1000)
%     'NumSteps'      optimizer parameter states (30)
%     'NumProbes'     fixed Rademacher probes per state (8)
%     'Rank'          exact-deflation target rank (100)
%     'SpectrumCount' absolute eigenvalues retained per tail (500)
%     'SolverKeys'    optional subset of registry keys (all)
%
%   The spectrum diagnostic is deliberately outside all timed solver totals.

    thisDir = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(fileparts(thisDir));
    addpath(repoRoot);
    addpath(thisDir);

    params = parse_options(thisDir, varargin{:});
    if ~exist(params.OutDir, 'dir'), mkdir(params.OutDir); end

    rng(params.Seed, 'twister');
    [X, y] = load_dataset_csv_or_mat(params.DataFile);
    if size(X, 2) ~= 4
        error('run_ard_training_benchmark:notCCPP', ...
              'This benchmark expects the four CCPP feature columns (found %d).', size(X, 2));
    end
    if params.NTrain + params.NTest > size(X, 1)
        error('run_ard_training_benchmark:tooManySamples', ...
              'NTrain + NTest exceeds the %d available rows.', size(X, 1));
    end

    perm = randperm(size(X, 1));
    trainIdx = perm(1:params.NTrain);
    testIdx = perm(params.NTrain + (1:params.NTest));
    [Xtr, ytr, Xte, yte, dataStats] = standardize_split( ...
        X(trainIdx, :), y(trainIdx), X(testIdx, :), y(testIdx));

    ell0 = coordinate_median_lengthscales(Xtr, params.Seed);
    theta0 = [log(ell0(:)); log(params.InitialSignalVariance); ...
              log(params.InitialNoiseVariance)];
    lower = [repmat(log(params.LengthscaleBounds(1)), 4, 1); ...
             log(params.SignalVarianceBounds(1)); log(params.NoiseVarianceBounds(1))];
    upper = [repmat(log(params.LengthscaleBounds(2)), 4, 1); ...
             log(params.SignalVarianceBounds(2)); log(params.NoiseVarianceBounds(2))];
    theta0 = min(max(theta0, lower), upper);

    % These common random numbers are shared by every independent training arm.
    probes = 2 * double(rand(params.NTrain, params.NumProbes) > 0.5) - 1;
    sketchOmega = randn(params.NTrain, params.SketchOversample * params.Rank);
    [~, ~, D2parts] = ard_rbf_kernel(Xtr, exp(theta0(1:4)));
    initialNlml = exact_gp_nlml(Xtr, ytr, theta0, D2parts);

    methods = method_registry(params);
    fprintf('[ARD GP] nTrain=%d nTest=%d states=%d probes=%d methods=%d\n', ...
            params.NTrain, params.NTest, params.NumSteps, params.NumProbes, numel(methods));
    fprintf('         initial ell=[%s], sf2=%.3g, sn2=%.3g, NLML/n=%.6g\n', ...
            sprintf(' %.3g', exp(theta0(1:4))), exp(theta0(5)), exp(theta0(6)), initialNlml);

    solveTables = cell(numel(methods), 1);
    trainTables = cell(numel(methods), 1);
    summaryRows = repmat(empty_summary(), numel(methods), 1);
    for im = 1:numel(methods)
        fprintf('\n========== method %d/%d: %s ==========\n', im, numel(methods), methods(im).key);
        % Reset before every arm so paired fresh/recycled exact eigs receive
        % the same initial random start and are directly comparable.
        rng(params.Seed + 500, 'twister');
        [solveTables{im}, trainTables{im}, summaryRows(im)] = train_one_method( ...
            methods(im), Xtr, ytr, Xte, yte, theta0, lower, upper, probes, ...
            sketchOmega, D2parts, initialNlml, params);
    end

    solveResults = vertcat(solveTables{:});
    trainingResults = vertcat(trainTables{:});
    summary = struct2table(summaryRows);
    writetable(solveResults, fullfile(params.OutDir, 'solve_results.csv'));
    writetable(trainingResults, fullfile(params.OutDir, 'training_results.csv'));
    writetable(summary, fullfile(params.OutDir, 'summary.csv'));

    cfg = struct('params', params, 'methods', methods, 'theta0', theta0, ...
                 'ell0', ell0, 'data_stats', dataStats, 'train_indices', trainIdx, ...
                 'test_indices', testIdx, 'initial_nlml_per_point', initialNlml, ...
                 'notes', {{'All methods share data, probes, initialization, and optimizer settings.', ...
                            'Each method drives an independent optimizer trajectory.', ...
                            'Spectrum and exact-quality diagnostics are excluded from timed totals.'}});
    save(fullfile(params.OutDir, 'run_config.mat'), 'cfg');
    write_json(fullfile(params.OutDir, 'run_config.json'), cfg);

    make_training_plots(solveResults, trainingResults, summary, params.OutDir);
    spectrum = plot_ard_training_spectra(Xtr, trainingResults, params);

    outputs = struct('solve_results', solveResults, 'training_results', trainingResults, ...
                     'summary', summary, 'spectrum', spectrum, 'config', cfg);
    save(fullfile(params.OutDir, 'benchmark_results.mat'), 'outputs', '-v7.3');
    fprintf('\n[ARD GP] complete: %s\n', params.OutDir);
end

% ========================================================================
function params = parse_options(thisDir, varargin)
    p = inputParser;
    p.addParameter('SmokeTest', false, @(x) islogical(x) || isnumeric(x));
    p.addParameter('DataFile', fullfile(thisDir, 'data', 'ccpp.csv'));
    p.addParameter('OutDir', '');
    p.addParameter('NTrain', 3000);
    p.addParameter('NTest', 1000);
    p.addParameter('NumSteps', 30);
    p.addParameter('NumProbes', 8);
    p.addParameter('Tol', 1e-6);
    p.addParameter('MaxIt', 10000);
    p.addParameter('Seed', 0);
    p.addParameter('Rank', 100);
    p.addParameter('Tau', 0.5);
    p.addParameter('SketchQList', [1 2 3]);
    p.addParameter('SketchOversample', 2);
    p.addParameter('AdamRate', 0.05);
    p.addParameter('SpectrumCount', 500);
    p.addParameter('SolverKeys', {});
    p.parse(varargin{:});
    params = p.Results;
    params.SmokeTest = logical(params.SmokeTest);
    params.InitialSignalVariance = 1;
    params.InitialNoiseVariance = 0.1;
    params.LengthscaleBounds = [0.05 20];
    params.SignalVarianceBounds = [1e-3 1e3];
    params.NoiseVarianceBounds = [1e-6 1];
    params.AdamBeta1 = 0.9;
    params.AdamBeta2 = 0.999;
    params.AdamEpsilon = 1e-8;
    params.icholOpts = struct('type', 'ict', 'droptol', 1e-3, 'michol', 'on');
    params.amgOpts = struct('maxLevels', 2, 'minCoarseSize', 800, ...
                            'theta', 0.05, 'omegaInterp', 0);
    if params.SmokeTest
        params.NTrain = min(params.NTrain, 300);
        params.NTest = min(params.NTest, 100);
        params.NumSteps = min(params.NumSteps, 3);
        params.NumProbes = min(params.NumProbes, 2);
        params.Rank = min(params.Rank, 20);
        params.MaxIt = min(params.MaxIt, 500);
    end
    if isempty(params.OutDir)
        if params.SmokeTest
            params.OutDir = fullfile(thisDir, 'benchmark_ard_training_smoke');
        else
            params.OutDir = fullfile(thisDir, 'benchmark_ard_training');
        end
    end
    params.OutDir = char(params.OutDir);
    params.DataFile = char(params.DataFile);
end

function methods = method_registry(params)
    methods = struct('key', {}, 'kind', {}, 'refresh', {}, 'q', {}, 'seed_offset', {});
    methods(end+1) = entry('unprec', 'unprec', 'fresh', 0);
    methods(end+1) = entry('ichol', 'ichol', 'fresh', 0);
    methods(end+1) = entry('amg', 'amg', 'fresh', 0);
    methods(end+1) = entry('defl_exact_recycle_once', 'defl_exact', 'recycle_once', 0);
    methods(end+1) = entry('defl_exact_fresh_oracle', 'defl_exact', 'fresh_oracle', 0);
    methods(end+1) = entry('twolevel_recycle_once', 'twolevel', 'recycle_once', 0);
    methods(end+1) = entry('twolevel_fresh_oracle', 'twolevel', 'fresh_oracle', 0);
    for q = params.SketchQList(:).'
        methods(end+1) = entry(sprintf('defl_sketch_q%d_recycle_once', q), ...
                               'defl_sketch', 'recycle_once', q); %#ok<AGROW>
        methods(end+1) = entry(sprintf('defl_sketch_q%d_fresh_oracle', q), ...
                               'defl_sketch', 'fresh_oracle', q); %#ok<AGROW>
    end
    for i = 1:numel(methods), methods(i).seed_offset = i; end
    if ~isempty(params.SolverKeys)
        requested = string(params.SolverKeys);
        keep = ismember(string({methods.key}), requested);
        missing = setdiff(requested, string({methods.key}));
        if ~isempty(missing)
            error('run_ard_training_benchmark:badSolverKey', ...
                  'Unknown SolverKeys: %s', strjoin(cellstr(missing), ', '));
        end
        methods = methods(keep);
    end
end

function m = entry(key, kind, refresh, q)
    m = struct('key', key, 'kind', kind, 'refresh', refresh, 'q', q, 'seed_offset', 0);
end

function [solveT, trainT, summary] = train_one_method(method, Xtr, ytr, Xte, yte, ...
        theta, lower, upper, probes, sketchOmega, D2parts, initialNlml, params)
    n = size(Xtr, 1);
    state = struct('V', []);
    adamM = zeros(size(theta));
    adamV = zeros(size(theta));
    previousA = [];
    previousKoff = [];
    solveCells = cell(params.NumSteps, 1);
    trainCells = cell(params.NumSteps, 1);
    failed = false;

    for step = 1:params.NumSteps
        ell = exp(theta(1:4)); sf2 = exp(theta(5)); sn2 = exp(theta(6));
        [K, dK, ~] = ard_rbf_kernel(Xtr, ell, D2parts);
        A = sf2 * K + sn2 * eye(n);
        Amul = @(Y) sf2 * (K * Y) + sn2 * Y;
        % The off-diagonal covariance includes signal variance; unlike the
        % diagonal noise term, every entry here changes under ARD/sf2 updates.
        Koff = sf2 * K; Koff(1:n+1:end) = 0;
        if isempty(previousA)
            matrixChange = NaN; offdiagChange = NaN;
        else
            matrixChange = norm(A - previousA, 'fro') / max(norm(previousA, 'fro'), eps);
            offdiagChange = norm(Koff - previousKoff, 'fro') / max(norm(previousKoff, 'fro'), eps);
        end

        tSetupAttempt = tic;
        try
            [Mfun, state, setup] = build_step_preconditioner( ...
                method, A, Amul, K, state, sketchOmega, step, params);
        catch ME
            setup = empty_setup(); setup.setup_time = toc(tSetupAttempt);
            warning('run_ard_training_benchmark:preconditionerFailed', ...
                    '%s step %d setup failed: %s', method.key, step, ME.message);
            rows = repmat(empty_solve_row(), size(probes, 2) + 1, 1);
            for j = 1:numel(rows)
                rows(j) = make_solve_row(method.key, step, j, NaN, -2, NaN, NaN, 0, setup);
            end
            solveCells{step} = struct2table(rows);
            trainCells{step} = struct2table(make_training_row(method.key, step, theta, ...
                nan(6, 1), matrixChange, offdiagChange, setup, false));
            failed = true;
            break;
        end
        rhs = [ytr, probes];
        Xsol = nan(size(rhs));
        rows = repmat(empty_solve_row(), size(rhs, 2), 1);
        stepFailed = false;
        for j = 1:size(rhs, 2)
            tSolve = tic;
            try
                if isempty(Mfun)
                    [x, flag, relres, iter] = pcg(Amul, rhs(:, j), params.Tol, params.MaxIt);
                else
                    [x, flag, relres, iter] = pcg(Amul, rhs(:, j), params.Tol, ...
                        params.MaxIt, Mfun, [], zeros(n, 1));
                end
                elapsed = toc(tSolve);
                trueRelres = norm(Amul(x) - rhs(:, j)) / norm(rhs(:, j));
            catch ME
                elapsed = toc(tSolve); x = nan(n, 1); flag = -1;
                relres = NaN; trueRelres = NaN; iter = NaN;
                warning('run_ard_training_benchmark:pcgFailed', ...
                        '%s step %d RHS %d: %s', method.key, step, j, ME.message);
            end
            Xsol(:, j) = x;
            rows(j) = make_solve_row(method.key, step, j, iter, flag, relres, ...
                                     trueRelres, elapsed, setup);
            stepFailed = stepFailed || flag ~= 0 || ~isfinite(trueRelres) || ...
                         trueRelres > 2 * params.Tol;
            fprintf('  step=%2d rhs=%s%02d it=%5g flag=%2d rel=%.2e\n', ...
                    step, ternary(j == 1, 'target', 'probe'), max(j-1, 0), ...
                    iter, flag, trueRelres);
        end
        solveCells{step} = struct2table(rows);

        if stepFailed
            grad = nan(6, 1);
            completed = false;
            failed = true;
        else
            grad = ard_gp_gradient(Xsol(:, 1), Xsol(:, 2:end), probes, ...
                                   K, dK, sf2, sn2);
            completed = all(isfinite(grad));
            failed = ~completed;
        end
        trainCells{step} = struct2table(make_training_row(method.key, step, theta, ...
            grad, matrixChange, offdiagChange, setup, completed));

        previousA = A;
        previousKoff = Koff;
        if failed, break; end
        if step < params.NumSteps
            adamM = params.AdamBeta1 * adamM + (1 - params.AdamBeta1) * grad;
            adamV = params.AdamBeta2 * adamV + (1 - params.AdamBeta2) * (grad.^2);
            mhat = adamM / (1 - params.AdamBeta1^step);
            vhat = adamV / (1 - params.AdamBeta2^step);
            theta = theta - params.AdamRate * mhat ./ (sqrt(vhat) + params.AdamEpsilon);
            theta = min(max(theta, lower), upper);
        end
    end

    solveCells = solveCells(~cellfun('isempty', solveCells));
    trainCells = trainCells(~cellfun('isempty', trainCells));
    solveT = vertcat(solveCells{:});
    trainT = vertcat(trainCells{:});
    finalTheta = [trainT.log_ell1(end); trainT.log_ell2(end); trainT.log_ell3(end); ...
                  trainT.log_ell4(end); trainT.log_signal_variance(end); ...
                  trainT.log_noise_variance(end)];
    [finalNlml, finalAlpha] = exact_gp_nlml(Xtr, ytr, finalTheta, D2parts);
    Ktest = exp(finalTheta(5)) * ard_rbf_cross_kernel(Xte, Xtr, exp(finalTheta(1:4)));
    prediction = Ktest * finalAlpha;
    rmse = sqrt(mean((prediction - yte).^2));

    summary = empty_summary();
    summary.method = string(method.key);
    summary.completed = ~failed && height(trainT) == params.NumSteps;
    summary.states_completed = sum(trainT.completed);
    summary.rhs_converged_fraction = mean(solveT.flag == 0 & solveT.true_relres <= 2 * params.Tol);
    summary.total_iterations = sum(solveT.iterations(isfinite(solveT.iterations)));
    summary.total_setup_time = sum(trainT.setup_time);
    summary.total_solve_time = sum(solveT.solve_time);
    summary.total_timed_time = summary.total_setup_time + summary.total_solve_time;
    summary.initial_nlml_per_point = initialNlml;
    summary.final_nlml_per_point = finalNlml;
    summary.heldout_standardized_rmse = rmse;
    summary.final_ell1 = exp(finalTheta(1)); summary.final_ell2 = exp(finalTheta(2));
    summary.final_ell3 = exp(finalTheta(3)); summary.final_ell4 = exp(finalTheta(4));
    summary.final_signal_variance = exp(finalTheta(5));
    summary.final_noise_variance = exp(finalTheta(6));
end

function [Mfun, state, setup] = build_step_preconditioner(method, A, Amul, K, ...
        state, sketchOmega, step, params)
    setup = empty_setup();
    tAll = tic;
    Mfun = [];
    rebuild = step == 1 || strcmp(method.refresh, 'fresh_oracle');

    switch method.kind
        case 'unprec'
            % No setup.

        case 'ichol'
            [L, ~, setup.ichol_build_time] = build_ichol_robust(A, params.icholOpts);
            Lt = L'; Mfun = @(r) Lt \ (L \ r);

        case 'amg'
            [L, ~, setup.ichol_build_time] = build_ichol_robust(A, params.icholOpts);
            Lt = L'; t = tic;
            Mfun = src.precond.make_amg_preconditioner(sparse(A), ...
                'maxLevels', params.amgOpts.maxLevels, ...
                'minCoarseSize', min(params.amgOpts.minCoarseSize, floor(size(A, 1)/4)), ...
                'theta', params.amgOpts.theta, 'omegaInterp', params.amgOpts.omegaInterp, ...
                'fineSmootherL', L, 'fineSmootherLt', Lt);
            setup.basis_build_time = toc(t);

        case 'defl_exact'
            if rebuild
                t = tic;
                [state.V, ~] = eigs(A, params.Rank, 'largestabs', 'Tolerance', 1e-8, ...
                                    'MaxIterations', 5000);
                [state.V, ~] = qr(real(state.V), 0);
                setup.basis_build_time = toc(t); setup.basis_rebuilt = true;
            end
            t = tic; Mfun = src.precond.deflation_P_apply(state.V, Amul, params.Tau);
            setup.coarse_build_time = toc(t);

        case 'defl_sketch'
            if rebuild
                t = tic;
                Y = src.precond.subspace_iter_plain(@(W) K * W, sketchOmega, method.q);
                [state.V, ~] = qr(Y, 0);
                setup.basis_build_time = toc(t); setup.basis_rebuilt = true;
            end
            t = tic; Mfun = src.precond.deflation_P_apply(state.V, Amul, params.Tau);
            setup.coarse_build_time = toc(t);

        case 'twolevel'
            [L, ~, setup.ichol_build_time] = build_ichol_robust(A, params.icholOpts);
            Lt = L'; Ahat = @(W) L \ (Amul(Lt \ W));
            if rebuild
                t = tic;
                [state.V, ~] = eigs(Ahat, size(A, 1), params.Rank, 'largestabs', ...
                    'IsFunctionSymmetric', true, 'Tolerance', 1e-6, 'MaxIterations', 5000);
                [state.V, ~] = qr(real(state.V), 0);
                setup.basis_build_time = toc(t); setup.basis_rebuilt = true;
            end
            t = tic;
            Papply = src.precond.deflation_P_apply(state.V, Ahat, params.Tau);
            Mfun = @(r) Lt \ Papply(L \ r);
            setup.coarse_build_time = toc(t);

        otherwise
            error('run_ard_training_benchmark:badMethodKind', 'Unknown kind: %s', method.kind);
    end
    setup.setup_time = toc(tAll);
end

function setup = empty_setup()
    setup = struct('setup_time', 0, 'ichol_build_time', 0, ...
                   'basis_build_time', 0, 'coarse_build_time', 0, ...
                   'basis_rebuilt', false);
end

function row = make_solve_row(method, step, rhsIndex, iterations, flag, relres, ...
        trueRelres, solveTime, setup)
    row = empty_solve_row();
    row.method = string(method); row.step = step;
    row.rhs_type = string(ternary(rhsIndex == 1, 'target', 'probe'));
    row.rhs_index = max(rhsIndex - 1, 0); row.iterations = iterations;
    row.flag = flag; row.reported_relres = relres; row.true_relres = trueRelres;
    row.solve_time = solveTime; row.step_setup_time = setup.setup_time;
end

function row = empty_solve_row()
    row = struct('method', "", 'step', NaN, 'rhs_type', "", 'rhs_index', NaN, ...
                 'iterations', NaN, 'flag', NaN, 'reported_relres', NaN, ...
                 'true_relres', NaN, 'solve_time', NaN, 'step_setup_time', NaN);
end

function row = make_training_row(method, step, theta, grad, matrixChange, ...
        offdiagChange, setup, completed)
    row = struct('method', string(method), 'step', step, ...
        'log_ell1', theta(1), 'log_ell2', theta(2), 'log_ell3', theta(3), ...
        'log_ell4', theta(4), 'log_signal_variance', theta(5), ...
        'log_noise_variance', theta(6), 'ell1', exp(theta(1)), ...
        'ell2', exp(theta(2)), 'ell3', exp(theta(3)), 'ell4', exp(theta(4)), ...
        'signal_variance', exp(theta(5)), 'noise_variance', exp(theta(6)), ...
        'grad_ell1', grad(1), 'grad_ell2', grad(2), 'grad_ell3', grad(3), ...
        'grad_ell4', grad(4), 'grad_signal_variance', grad(5), ...
        'grad_noise_variance', grad(6), 'gradient_norm', norm(grad), ...
        'matrix_relative_change', matrixChange, ...
        'offdiag_kernel_relative_change', offdiagChange, ...
        'setup_time', setup.setup_time, 'ichol_build_time', setup.ichol_build_time, ...
        'basis_build_time', setup.basis_build_time, ...
        'coarse_build_time', setup.coarse_build_time, ...
        'basis_rebuilt', setup.basis_rebuilt, 'completed', completed);
end

function row = empty_summary()
    row = struct('method', "", 'completed', false, 'states_completed', 0, ...
        'rhs_converged_fraction', NaN, 'total_iterations', NaN, ...
        'total_setup_time', NaN, 'total_solve_time', NaN, 'total_timed_time', NaN, ...
        'initial_nlml_per_point', NaN, 'final_nlml_per_point', NaN, ...
        'heldout_standardized_rmse', NaN, 'final_ell1', NaN, 'final_ell2', NaN, ...
        'final_ell3', NaN, 'final_ell4', NaN, 'final_signal_variance', NaN, ...
        'final_noise_variance', NaN);
end

function [nlml, alpha] = exact_gp_nlml(X, y, theta, D2parts)
    K = ard_rbf_kernel(X, exp(theta(1:4)), D2parts);
    A = exp(theta(5)) * K + exp(theta(6)) * eye(size(X, 1));
    L = chol((A + A') / 2, 'lower');
    alpha = L' \ (L \ y);
    nlml = (0.5 * (y' * alpha) + sum(log(diag(L))) + ...
            0.5 * numel(y) * log(2*pi)) / numel(y);
end

function [Xtr, ytr, Xte, yte, stats] = standardize_split(Xtr0, ytr0, Xte0, yte0)
    muX = mean(Xtr0, 1); sigmaX = std(Xtr0, 0, 1); sigmaX(sigmaX < eps) = 1;
    muy = mean(ytr0); sigmay = std(ytr0); if sigmay < eps, sigmay = 1; end
    Xtr = (Xtr0 - muX) ./ sigmaX; Xte = (Xte0 - muX) ./ sigmaX;
    ytr = (ytr0 - muy) / sigmay; yte = (yte0 - muy) / sigmay;
    stats = struct('mu_X', muX, 'sigma_X', sigmaX, 'mu_y', muy, 'sigma_y', sigmay);
end

function ell = coordinate_median_lengthscales(X, seed)
    rng(seed + 991, 'twister');
    m = min(size(X, 1), 1000); idx = randperm(size(X, 1), m);
    ell = zeros(size(X, 2), 1);
    mask = triu(true(m), 1);
    for r = 1:size(X, 2)
        D = abs(X(idx, r) - X(idx, r).');
        ell(r) = median(D(mask));
        if ~(ell(r) > 0 && isfinite(ell(r))), ell(r) = 1; end
    end
end

function make_training_plots(solveT, trainT, summary, outDir)
    methods = unique(trainT.method, 'stable'); colors = lines(numel(methods));
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1100 600]); hold on;
    for i = 1:numel(methods)
        key = methods(i); steps = unique(solveT.step(solveT.method == key)); totals = nan(size(steps));
        for j = 1:numel(steps)
            mask = solveT.method == key & solveT.step == steps(j);
            totals(j) = sum(solveT.iterations(mask), 'omitnan');
        end
        plot(steps, totals, '-o', 'Color', colors(i,:), 'DisplayName', char(key));
    end
    grid on; xlabel('optimizer state'); ylabel('total PCG iterations over RHSs');
    legend('Location', 'eastoutside', 'Interpreter', 'none');
    title('ARD GP training: linear-solve work per state');
    exportgraphics(fig, fullfile(outDir, 'iterations_per_step.png'), 'Resolution', 180); close(fig);

    canonical = "defl_exact_fresh_oracle";
    C = trainT(trainT.method == canonical & trainT.completed, :);
    if ~isempty(C)
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1000 600]);
        semilogy(C.step, [C.ell1 C.ell2 C.ell3 C.ell4 C.signal_variance C.noise_variance], ...
                 '-o', 'LineWidth', 1.2); grid on; xlabel('optimizer state'); ylabel('parameter value');
        legend({'ell1','ell2','ell3','ell4','signal variance','noise variance'}, ...
               'Location', 'eastoutside'); title('Canonical ARD hyperparameter trajectory');
        exportgraphics(fig, fullfile(outDir, 'hyperparameter_trajectories.png'), 'Resolution', 180); close(fig);

        fig = figure('Visible', 'off', 'Color', 'w');
        semilogy(C.step, max(C.offdiag_kernel_relative_change, eps), '-o', 'LineWidth', 1.3);
        grid on; xlabel('optimizer state'); ylabel('relative off-diagonal kernel change');
        title('Canonical matrix-sequence drift');
        exportgraphics(fig, fullfile(outDir, 'matrix_change.png'), 'Resolution', 180); close(fig);
    end

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 900 600]);
    scatter(summary.total_timed_time, summary.final_nlml_per_point, 55, 'filled'); grid on;
    xlabel('timed setup + solve seconds'); ylabel('final exact NLML per point');
    title('Training quality versus linear-algebra time');
    for i = 1:height(summary)
        text(summary.total_timed_time(i), summary.final_nlml_per_point(i), ...
             ['  ' char(summary.method(i))], 'Interpreter', 'none', 'FontSize', 7);
    end
    exportgraphics(fig, fullfile(outDir, 'quality_vs_runtime.png'), 'Resolution', 180); close(fig);
end

function write_json(path, value)
    fid = fopen(path, 'w');
    if fid < 0, error('run_ard_training_benchmark:jsonOpen', 'Cannot open %s', path); end
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(value, 'PrettyPrint', true), 'char');
end

function out = ternary(condition, yes, no)
    if condition, out = yes; else, out = no; end
end
