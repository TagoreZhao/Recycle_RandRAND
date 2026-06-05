function resultsTable = run_logistic_benchmark(varargin)
%RUN_LOGISTIC_BENCHMARK  Fit logistic regression and inspect the Newton Hessian.
%   RUN_LOGISTIC_BENCHMARK() loads the LIBSVM dataset in this folder's data/
%   directory, fits L2-penalized logistic regression by matrix-free Newton-CG
%   for each lambda in a sweep, and records the eigenspectrum of the converged
%   Newton Hessian H = Xa'WXa + lambda*I. The goal is to see whether H exhibits
%   a "spiking small spectrum" (a cluster of eigenvalues pinned at the lambda
%   floor), which occurs when rank(Xa'WXa) < d (the n <= d regime).
%
%   resultsTable = RUN_LOGISTIC_BENCHMARK(...) also returns the results table
%   and saves results/logistic_hessian_results.{mat,csv} plus
%   results/figures/hessian_eig_spectrum.pdf.
%
%   Name-value options (all optional):
%     'DataFile'   : LIBSVM file path (default: auto-detect in data/).
%     'LambdaList' : L2 penalties to sweep (default logspace(-4, 2, 13)).
%     'Seed'       : RNG seed (default 0).
%     'MaxIter'    : Newton iterations cap (default 100).
%     'Tol'        : Newton relative grad-norm tolerance (default 1e-8).
%     'K'          : top-k eigenvalues when dim > DenseMax (default 400).
%     'DenseMax'   : dense full-eig threshold on d+1 (default 8000).
%     'OutDir'     : output directory (default results).
%
%   Example
%     T = run_logistic_benchmark('LambdaList', logspace(-3, 1, 5));
%
%   See also load_libsvm, standardize_features, logreg_newton,
%   hessian_spectrum, plot_hessian_spectrum.

    thisDir = fileparts(mfilename('fullpath'));

    p = inputParser;
    p.addParameter('DataFile', '');
    p.addParameter('LambdaList', logspace(-4, 2, 13));
    p.addParameter('Seed', 0);
    p.addParameter('MaxIter', 100);
    p.addParameter('Tol', 1e-8);
    p.addParameter('K', 500);
    p.addParameter('DenseMax', 8000);
    p.addParameter('OutDir', fullfile(thisDir, 'results'));
    p.parse(varargin{:});
    opt = p.Results;

    dataFile = opt.DataFile;
    if isempty(dataFile)
        dataFile = autodetect_data(fullfile(thisDir, 'data'));
    end
    if ~exist(opt.OutDir, 'dir'), mkdir(opt.OutDir); end

    [~, dsName] = fileparts(thisDir);

    % --- Load + sparse-safe standardize ---
    rng(opt.Seed);
    [X, y, info] = load_libsvm(dataFile);
    [Xs, ~, keep] = standardize_features(X);
    n = size(Xs, 1);
    d = size(Xs, 2);
    fprintf('Loaded %s: n=%d, d=%d (kept %d/%d cols), labels=[%g %g]\n', ...
            dataFile, n, d, numel(keep), info.d, info.orig_labels(1), ...
            info.orig_labels(2));

    lambdas = opt.LambdaList(:).';
    nL = numel(lambdas);
    newtonOpts = struct('MaxIter', opt.MaxIter, 'Tol', opt.Tol);
    specOpts   = struct('DenseMax', opt.DenseMax, 'K', opt.K);

    rows = repmat(emptyRow(), 1, nL);
    specByLambda = repmat(struct('eigs_small', [], 'eigs_large', [], 'mode', "", ...
        'min', NaN, 'max', NaN, 'cond', NaN, 'lambda_floor', NaN, ...
        'n_at_floor', NaN), 1, nL);

    for il = 1:nL
        lam = lambdas(il);
        fprintf('\n=== lambda=%.4g (%d/%d) ===\n', lam, il, nL);

        [~, ninfo, w] = logreg_newton(Xs, y, lam, newtonOpts);
        spec = hessian_spectrum(Xs, w, lam, specOpts);
        specByLambda(il) = spec;

        rows(il).lambda          = lam;
        rows(il).n               = n;
        rows(il).d               = d + 1;          % includes intercept
        rows(il).newton_iters    = ninfo.newton_iters;
        rows(il).cg_iters_total  = ninfo.cg_iters_total;
        rows(il).converged       = ninfo.converged;
        rows(il).train_acc       = ninfo.train_acc;
        rows(il).min_eig         = spec.min;
        rows(il).max_eig         = spec.max;
        rows(il).cond            = spec.cond;
        rows(il).n_at_floor      = spec.n_at_floor;
        rows(il).spectrum_mode   = spec.mode;
        rows(il).separable       = ninfo.separable;

        fprintf(['  newton_it=%d cg=%d acc=%.3f | %s spectrum: ', ...
                 'min=%.3e max=%.3e cond=%.2e n_at_floor=%d\n'], ...
                ninfo.newton_iters, ninfo.cg_iters_total, ninfo.train_acc, ...
                spec.mode, spec.min, spec.max, spec.cond, spec.n_at_floor);
    end

    resultsTable = struct2table(rows);

    matFile = fullfile(opt.OutDir, 'logistic_hessian_results.mat');
    csvFile = fullfile(opt.OutDir, 'logistic_hessian_results.csv');
    meta = struct('dataset', dsName, 'data_file', dataFile, ...
                  'lambda_list', lambdas, 'n', n, 'd', d + 1, ...
                  'd_original', info.d, 'cols_kept', numel(keep), ...
                  'seed', opt.Seed, 'dense_max', opt.DenseMax, 'topk', opt.K);
    save(matFile, 'resultsTable', 'meta', 'specByLambda');
    writetable(resultsTable, csvFile);
    fprintf('\nSaved results: %s\n                %s\n', matFile, csvFile);

    plot_hessian_spectrum(specByLambda, lambdas, ...
                          fullfile(opt.OutDir, 'figures'), dsName);
    fprintf('Done.\n');
end

%% --------- local helpers ---------
function r = emptyRow()
%EMPTYROW  Prototype results row (defines column order/types).
    r = struct('lambda', NaN, 'n', NaN, 'd', NaN, 'newton_iters', NaN, ...
               'cg_iters_total', NaN, 'converged', false, 'train_acc', NaN, ...
               'min_eig', NaN, 'max_eig', NaN, 'cond', NaN, ...
               'n_at_floor', NaN, 'spectrum_mode', "", 'separable', false);
end

function f = autodetect_data(dataDir)
%AUTODETECT_DATA  Pick the single uncompressed data file in dataDir.
    if ~exist(dataDir, 'dir')
        error('run_logistic_benchmark:noData', ...
              'No data/ directory. Run the download_*.m script first.');
    end
    listing = dir(dataDir);
    cand = {};
    for k = 1:numel(listing)
        if listing(k).isdir, continue; end
        nm = listing(k).name;
        if endsWith(nm, '.bz2') || startsWith(nm, '.'), continue; end
        cand{end + 1} = fullfile(dataDir, nm); %#ok<AGROW>
    end
    if isempty(cand)
        error('run_logistic_benchmark:noData', ...
              'No uncompressed data file found in %s. Run download_*.m.', dataDir);
    end
    if numel(cand) > 1
        warning('run_logistic_benchmark:multipleData', ...
                'Multiple data files in %s; using %s.', dataDir, cand{1});
    end
    f = cand{1};
end
