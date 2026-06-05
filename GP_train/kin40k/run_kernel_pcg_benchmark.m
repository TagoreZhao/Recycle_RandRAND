function resultsTable = run_kernel_pcg_benchmark(varargin)
%RUN_KERNEL_PCG_BENCHMARK  Benchmark PCG vs PCG+ichol on kernel-ridge systems.
%   RUN_KERNEL_PCG_BENCHMARK() runs the full benchmark on the UCI Elevators
%   dataset: for each subset size it sweeps the RBF lengthscale over 20
%   values, building SPD systems A_j = K_{ell_j}(X,X) + sigma2*I, and solves
%   each with MATLAB's pcg twice (no preconditioner, and incomplete-Cholesky).
%   Results are saved to results/ as .mat and .csv, and plotted.
%
%   resultsTable = RUN_KERNEL_PCG_BENCHMARK(...) also returns the results
%   table.
%
%   Name-value options (all optional; defaults follow the spec):
%     'DataFile'     : path to .csv/.mat dataset (default data/elevators.csv).
%     'SubsetSizes'  : vector of subset sizes (default [3000 5000 10000 16599]).
%     'CList'        : lengthscale multipliers (default linspace(0.5, 2.0, 20)).
%     'Sigma2'       : ridge term (default 1e-3).
%     'Tol'          : PCG tolerance (default 1e-8).
%     'MaxIt'        : PCG max iterations (default 1000).
%     'Seed'         : RNG seed (default 0).
%     'OutDir'       : output directory (default results).
%
%   Example
%     % Quick smoke run:
%     T = run_kernel_pcg_benchmark('SubsetSizes', [300 500], ...
%                                  'CList', linspace(0.5, 2.0, 4));
%
%   See also load_dataset_csv_or_mat, standardize_data,
%   estimate_median_lengthscale, rbf_kernel_matrix, run_pcg_sequence,
%   plot_pcg_results.

    thisDir = fileparts(mfilename('fullpath'));

    p = inputParser;
    p.addParameter('DataFile', fullfile(thisDir, 'data', 'kin40k.csv'));
    p.addParameter('SubsetSizes', [3000, 5000, 10000, 20000]);
    p.addParameter('CList', linspace(0.5, 2.0, 20));
    p.addParameter('Sigma2', 1e-3);
    p.addParameter('Tol', 1e-8);
    p.addParameter('MaxIt', 1000);
    p.addParameter('Seed', 0);
    p.addParameter('OutDir', fullfile(thisDir, 'results'));
    p.parse(varargin{:});
    opt = p.Results;

    icholOpts = struct('type', 'ict', 'droptol', 1e-3, 'michol', 'on');

    if ~exist(opt.OutDir, 'dir')
        mkdir(opt.OutDir);
    end

    % --- Load + standardize (fixed seed for reproducibility) ---
    rng(opt.Seed);
    [X, y] = load_dataset_csv_or_mat(opt.DataFile);
    [Xs, ys] = standardize_data(X, y);
    N = size(Xs, 1);
    fprintf('Loaded %d samples, %d features from %s\n', N, size(Xs, 2), opt.DataFile);

    % Fixed permutation once; nested first-n subsets.
    perm = randperm(N);

    allRows = [];
    for s = 1:numel(opt.SubsetSizes)
        n = opt.SubsetSizes(s);
        if n > N
            fprintf('Skipping subset size %d (> dataset size %d).\n', n, N);
            continue;
        end
        idx = perm(1:n);
        Xn = Xs(idx, :);
        yn = ys(idx);

        ell0 = estimate_median_lengthscale(Xn);
        fprintf('\n=== Subset n=%d | base lengthscale ell0=%.4g ===\n', n, ell0);

        rows = run_pcg_sequence(Xn, yn, opt.CList, ell0, ...
                                opt.Sigma2, opt.Tol, opt.MaxIt, icholOpts);
        allRows = [allRows, rows]; %#ok<AGROW>
    end

    if isempty(allRows)
        error('run_kernel_pcg_benchmark:noResults', ...
              'No subset sizes were small enough to run.');
    end

    resultsTable = struct2table(allRows);

    matFile = fullfile(opt.OutDir, 'kernel_pcg_results.mat');
    csvFile = fullfile(opt.OutDir, 'kernel_pcg_results.csv');
    meta = struct('subset_sizes', opt.SubsetSizes, 'c_list', opt.CList, ...
                  'sigma2', opt.Sigma2, 'tol', opt.Tol, 'maxit', opt.MaxIt, ...
                  'seed', opt.Seed, 'ichol_opts', icholOpts, ...
                  'data_file', opt.DataFile, 'n_total', N);
    save(matFile, 'resultsTable', 'meta');
    writetable(resultsTable, csvFile);
    fprintf('\nSaved results: %s\n                %s\n', matFile, csvFile);

    % --- Plots ---
    plot_pcg_results(resultsTable, fullfile(opt.OutDir, 'figures'));
    fprintf('Done.\n');
end
