function resultsTable = run_logistic_benchmark(varargin)
%RUN_LOGISTIC_BENCHMARK  Fit logistic regression and inspect the Newton Hessian.
%   RUN_LOGISTIC_BENCHMARK() loads the LIBSVM dataset in this folder's data/
%   directory, fits L2-penalized logistic regression by matrix-free Newton-CG
%   for each lambda in a sweep, and records the eigenspectrum of the converged
%   Newton Hessian H = Xa'WXa + lambda*I. The goal is to see whether H exhibits
%   a "spiking small spectrum" (a cluster of eigenvalues pinned at the lambda
%   floor), which occurs when rank(Xa'WXa) < d (the n <= d regime).
%
%   Three spectrum figures are produced:
%     hessian_eig_spectrum.pdf          : one curve per lambda, each showing
%                                         the CONVERGED (last) Newton Hessian
%                                         of that lambda's sequence.
%     hessian_eig_spectrum_first.pdf    : one curve per lambda, each showing
%                                         the FIRST Newton system. Since
%                                         Newton starts at beta = 0, this is
%                                         H_1 = 0.25*Xa'Xa + lambda*I for
%                                         every lambda, so all curves are
%                                         exact lambda-shifts of ONE base
%                                         spectrum (computed once).
%     hessian_eig_spectrum_sequence.pdf : for the single lambda selected by
%                                         'SeqLambda', one curve per Newton
%                                         iteration k, i.e. the full sequence
%                                         of linear systems
%                                         H_k = Xa'W_kXa + lambda*I.
%
%   resultsTable = RUN_LOGISTIC_BENCHMARK(...) also returns the results table
%   and saves results/logistic_hessian_results.{mat,csv} plus the figures
%   above under results/figures/.
%
%   Name-value options (all optional):
%     'DataFile'   : LIBSVM file path (default: auto-detect in data/).
%     'LambdaList' : L2 penalties to sweep (default logspace(-4, 2, 13)).
%     'Seed'       : RNG seed (default 0).
%     'MaxIter'    : Newton iterations cap (default 100).
%     'Tol'        : Newton relative grad-norm tolerance (default 1e-8).
%     'K'          : top-k eigenvalues when dim > DenseMax (default 400).
%     'DenseMax'   : dense full-eig threshold on d+1 (default 8000).
%     'SeqLambda'  : lambda whose Newton sequence gets per-iteration spectra
%                    (default 1e-2; snapped to the nearest LambdaList entry).
%                    Pass [] to skip the sequence figure.
%     'OutDir'     : output directory (default results).
%
%   Example
%     T = run_logistic_benchmark('LambdaList', logspace(-3, 1, 5));
%
%   See also load_libsvm, standardize_features, logreg_newton,
%   hessian_spectrum, plot_hessian_spectrum, plot_hessian_sequence_spectrum.

    thisDir = fileparts(mfilename('fullpath'));

    p = inputParser;
    p.addParameter('DataFile', '');
    p.addParameter('LambdaList', logspace(-4, 2, 13));
    p.addParameter('Seed', 0);
    p.addParameter('MaxIter', 100);
    p.addParameter('Tol', 1e-8);
    p.addParameter('K', 500);
    p.addParameter('DenseMax', 8000);
    p.addParameter('SeqLambda', 1e-2);
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

    % Snap SeqLambda to the nearest swept lambda (its Newton sequence gets
    % per-iteration spectra); [] disables the sequence figure.
    seqIdx = [];
    if ~isempty(opt.SeqLambda)
        [~, seqIdx] = min(abs(lambdas - opt.SeqLambda));
        if lambdas(seqIdx) ~= opt.SeqLambda
            warning('run_logistic_benchmark:seqLambdaSnap', ...
                    'SeqLambda=%.4g not in LambdaList; using nearest %.4g.', ...
                    opt.SeqLambda, lambdas(seqIdx));
        end
    end
    seqLambda  = [];
    specByIter = [];

    rows = repmat(emptyRow(), 1, nL);
    specByLambda = repmat(struct('eigs_small', [], 'eigs_large', [], ...
        'eigs_full', [], 'mode', "", 'min', NaN, 'max', NaN, 'cond', NaN, ...
        'lambda_floor', NaN, 'n_at_floor', NaN), 1, nL);

    % First Newton system of EVERY lambda: beta = 0 gives w = 0.25, so
    % H_1(lambda) = 0.25*Xa'Xa + lambda*I. One spectrum at lambdas(1), then
    % exact lambda-shifts (never computed at lambda = 0, which would make the
    % matrix-free inverse handle singular).
    fprintf('\nFirst-system base spectrum at lambda=%.4g ...\n', lambdas(1));
    w1 = 0.25 * ones(n, 1);
    specBase = hessian_spectrum(Xs, w1, lambdas(1), specOpts);
    specFirstByLambda = repmat(specBase, 1, nL);
    for il = 1:nL
        specFirstByLambda(il) = shift_spectrum(specBase, ...
                                    lambdas(il) - lambdas(1), lambdas(il));
    end

    for il = 1:nL
        lam = lambdas(il);
        fprintf('\n=== lambda=%.4g (%d/%d) ===\n', lam, il, nL);

        [~, ninfo, w] = logreg_newton(Xs, y, lam, newtonOpts);
        spec = hessian_spectrum(Xs, w, lam, specOpts);
        specByLambda(il) = spec;

        if isequal(il, seqIdx)
            % Spectrum of every system H_k = Xa'diag(w_k)Xa + lam*I in this
            % lambda's Newton sequence (column k of w_history = weights of
            % the k-th solve; the first system has w = 0.25 everywhere).
            seqLambda = lam;
            nIt = size(ninfo.w_history, 2);
            fprintf('  sequence spectra: %d Newton systems at lambda=%.4g\n', ...
                    nIt, lam);
            specByIter = repmat(spec, 1, nIt);   % prototype for struct layout
            for k = 1:nIt
                specByIter(k) = hessian_spectrum(Xs, ninfo.w_history(:, k), ...
                                                 lam, specOpts);
            end
        end

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
                  'seed', opt.Seed, 'dense_max', opt.DenseMax, 'topk', opt.K, ...
                  'seq_lambda', seqLambda);
    save(matFile, 'resultsTable', 'meta', 'specByLambda', ...
         'specFirstByLambda', 'specByIter');
    writetable(resultsTable, csvFile);
    fprintf('\nSaved results: %s\n                %s\n', matFile, csvFile);

    figDir = fullfile(opt.OutDir, 'figures');
    plot_hessian_spectrum(specByLambda, lambdas, figDir, dsName);
    plot_hessian_spectrum(specFirstByLambda, lambdas, figDir, dsName, ...
                          'hessian_eig_spectrum_first.pdf', ...
                          'first Newton system (\beta=0, W=0.25I)');
    if ~isempty(specByIter)
        plot_hessian_sequence_spectrum(specByIter, seqLambda, figDir, dsName);
    end
    fprintf('Done.\n');
end

%% --------- local helpers ---------
function spec = shift_spectrum(specBase, dlam, lamNew)
%SHIFT_SPECTRUM  Exact spectrum of H0 + dlam*I from the spectrum of H0.
%   Adding a multiple of the identity shifts every eigenvalue by dlam, so the
%   first-system spectra of the whole lambda sweep follow from ONE base
%   spectrum. n_at_floor is recounted against the new floor lamNew using the
%   full spectrum when available (dense mode); otherwise the smallest-K slice
%   caps the count at K.
    spec = specBase;
    spec.eigs_small   = specBase.eigs_small + dlam;
    spec.eigs_large   = specBase.eigs_large + dlam;
    spec.eigs_full    = specBase.eigs_full + dlam;   % [] + dlam stays []
    spec.min          = specBase.min + dlam;
    spec.max          = specBase.max + dlam;
    spec.cond         = spec.max / spec.min;
    spec.lambda_floor = lamNew;
    if isempty(spec.eigs_full)
        spec.n_at_floor = sum(spec.eigs_small <= 1.01 * lamNew);
    else
        spec.n_at_floor = sum(spec.eigs_full <= 1.01 * lamNew);
    end
end

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
