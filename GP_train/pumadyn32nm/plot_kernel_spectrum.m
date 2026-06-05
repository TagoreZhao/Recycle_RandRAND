function plot_kernel_spectrum(varargin)
%PLOT_KERNEL_SPECTRUM  Eigen-spectrum overlay of kernel-ridge systems vs ichol.
%   PLOT_KERNEL_SPECTRUM() builds the dense kernel-ridge matrices
%   A = K_{c*ell0}(X,X) + sigma2*I for several lengthscale multipliers c on a
%   fixed subset of the dataset, and overlays the smallest-k and largest-k
%   eigenvalues of A against those of the incomplete-Cholesky preconditioned
%   operator M = L^{-1} A L^{-T}. This visualizes how the lengthscale changes
%   the conditioning of A and whether ichol clusters the spectrum.
%
%   PLOT_KERNEL_SPECTRUM(NAME, VALUE, ...) accepts options:
%     'DataFile' : dataset path (default data/elevators.csv).
%     'N'        : subset size (default 2000).
%     'CList'    : lengthscale multipliers to overlay (default [0.5 1.0 2.0]).
%     'K'        : eigenvalues per end (default 200; clamped to floor(N/2)-1).
%     'Sigma2'   : ridge term (default 1e-3).
%     'Seed'     : RNG seed (default 0).
%     'OutDir'   : output directory (default results/figures).
%
%   Example
%     plot_kernel_spectrum('N', 2000, 'CList', [0.5 1.0 2.0]);
%
%   See also rbf_kernel_matrix, build_ichol_robust, run_kernel_pcg_benchmark.
%
%   Mirrors the eigen-spectrum diagnostics in
%   subspace_capture/run_subspace_capture.m, adapted to dense kernel systems.

    thisDir = fileparts(mfilename('fullpath'));
    addpath(thisDir);   % resolve sibling helper functions

    p = inputParser;
    p.addParameter('DataFile', fullfile(thisDir, 'data', 'pumadyn32nm.csv'));
    p.addParameter('N', 2000);
    p.addParameter('CList', [0.5, 1.0, 2.0]);
    p.addParameter('K', 200);
    p.addParameter('Sigma2', 1e-3);
    p.addParameter('Seed', 0);
    p.addParameter('OutDir', fullfile(thisDir, 'results', 'figures'));
    p.parse(varargin{:});
    opt = p.Results;

    if ~exist(opt.OutDir, 'dir')
        mkdir(opt.OutDir);
    end

    icholOpts = struct('type', 'ict', 'droptol', 1e-3, 'michol', 'on');

    % --- Load + standardize + nested first-n subset (benchmark convention) ---
    rng(opt.Seed);
    [X, y] = load_dataset_csv_or_mat(opt.DataFile);
    [Xs, ~] = standardize_data(X, y);
    Ntot = size(Xs, 1);
    N = min(opt.N, Ntot);
    perm = randperm(Ntot);
    Xn = Xs(perm(1:N), :);

    k = min(opt.K, floor(N / 2) - 1);
    if k < opt.K
        warning('plot_kernel_spectrum:clampK', ...
                'K reduced from %d to %d (must be < N/2 for N=%d).', opt.K, k, N);
    end

    ell0 = estimate_median_lengthscale(Xn);
    cList = opt.CList(:).';
    nC = numel(cList);
    fprintf('Subset n=%d, base lengthscale ell0=%.4g, k=%d eigenvalues/end\n', ...
            N, ell0, k);

    % --- Per-c eigen-spectra of A and M = L^{-1} A L^{-T} ---
    res = repmat(struct('c', NaN, 'ell', NaN, ...
                        'A_small', [], 'A_large', [], ...
                        'M_small', [], 'M_large', [], ...
                        'condA', NaN, 'condM', NaN), 1, nC);
    for ic = 1:nC
        c   = cList(ic);
        ell = c * ell0;
        A = rbf_kernel_matrix(Xn, ell) + opt.Sigma2 * eye(N);
        L = build_ichol_robust(A, icholOpts);

        [aS, aL] = eig_ends_explicit(A, k);
        [mS, mL] = eig_ends_precond(A, L, k);

        res(ic).c       = c;
        res(ic).ell     = ell;
        res(ic).A_small = aS;  res(ic).A_large = aL;
        res(ic).M_small = mS;  res(ic).M_large = mL;
        res(ic).condA   = aL(1) / aS(1);
        res(ic).condM   = mL(1) / mS(1);

        fprintf(['  c=%.3f ell=%.4g | A: lam_min=%.3e lam_max=%.3e cond=%.2e' ...
                 ' | M: lam_min=%.3e lam_max=%.3e cond=%.2e\n'], ...
                c, ell, aS(1), aL(1), res(ic).condA, ...
                mS(1), mL(1), res(ic).condM);
    end

    % --- Render overlay figure (smallest | largest) ---
    fig = figure('Color', 'w', 'Position', [50, 50, 1300, 620]);
    tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    draw_spectrum_overlay(nexttile(tl), res, 'smallest', k);
    draw_spectrum_overlay(nexttile(tl), res, 'largest', k);

    title(tl, sprintf('Kernel-ridge spectrum: A vs M = L^{-1} A L^{-T}  (n=%d, \\sigma^2=%g)', ...
          N, opt.Sigma2), 'FontWeight', 'bold', 'FontSize', 13, 'Interpreter', 'tex');
    subtitle(tl, cond_summary(res), 'Interpreter', 'tex', 'FontSize', 10);

    outFile = fullfile(opt.OutDir, 'kernel_eig_spectrum.pdf');
    exportgraphics(fig, outFile, 'ContentType', 'vector');
    fprintf('Saved %s\n', outFile);
    close(fig);
end

%% =========================================================================
%% Local helpers
%% =========================================================================
function [d_small, d_large] = eig_ends_explicit(A, k)
%EIG_ENDS_EXPLICIT  Smallest-k and largest-k eigenvalues of explicit SPD A.
%   Returns column vectors sorted ascending (d_small) and descending (d_large).
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    dS = eigs(A, k, 'smallestabs', opts);
    dL = eigs(A, k, 'largestabs', opts);
    d_small = sort(real(dS), 'ascend');
    d_large = sort(real(dL), 'descend');
end

function [d_small, d_large] = eig_ends_precond(A, L, k)
%EIG_ENDS_PRECOND  Smallest-k and largest-k eigenvalues of M = L^{-1} A L^{-T}.
%   Largest via the forward handle Tfun = L\(A*(L'\x)); smallest via the
%   inverse handle Tinv = L'*(A\(L*x)) so 'smallestabs' becomes a largest
%   problem on M^{-1} (mirrors run_subspace_capture.m). M is symmetric.
    n  = size(A, 1);
    Lt = L';

    optsL = struct('Tolerance', 1e-10, 'MaxIterations', 5000, ...
                   'IsFunctionSymmetric', true);
    Tfun = @(x) L \ (A * (Lt \ x));
    dL   = eigs(Tfun, n, k, 'largestabs', optsL);
    d_large = sort(real(dL), 'descend');

    dA   = decomposition(A, 'chol');
    Tinv = @(x) Lt * (dA \ (L * x));
    optsS = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    dS   = eigs(Tinv, n, k, 'smallestabs', optsS);
    d_small = sort(real(dS), 'ascend');
end

function draw_spectrum_overlay(ax, res, mode, k)
%DRAW_SPECTRUM_OVERLAY  Overlay A (warm) and M (cool) spectra across c values.
%   mode = 'smallest' : ascending eigenvalues, x-axis reversed (k -> 1).
%   mode = 'largest'  : descending eigenvalues, natural x-axis (1 -> k).
%   Darker shade = larger c (smoother kernel).
    nC = numel(res);

    % Light -> dark ramps: warm for A, cool for M.
    A_light = [0.97 0.78 0.74];   A_dark = [0.55 0.10 0.08];
    M_light = [0.74 0.84 0.95];   M_dark = [0.06 0.22 0.50];

    hold(ax, 'on'); grid(ax, 'on');
    legH = gobjects(2 * nC, 1);
    legL = cell(2 * nC, 1);

    for ic = 1:nC
        t  = ramp_t(ic, nC);
        cA = (1 - t) * A_light + t * A_dark;
        cM = (1 - t) * M_light + t * M_dark;

        if strcmp(mode, 'smallest')
            DA = res(ic).A_small(:);
            DM = res(ic).M_small(:);
        else
            DA = res(ic).A_large(:);
            DM = res(ic).M_large(:);
        end
        DA(DA <= 0) = NaN;
        DM(DM <= 0) = NaN;

        hA = semilogy(ax, 1:numel(DA), DA, '-', 'LineWidth', 1.6, 'Color', cA);
        hM = semilogy(ax, 1:numel(DM), DM, '-', 'LineWidth', 1.6, 'Color', cM);

        legH(2*ic - 1) = hA;
        legH(2*ic)     = hM;
        legL{2*ic - 1} = sprintf('A   (c=%.3g)', res(ic).c);
        legL{2*ic}     = sprintf('M   (c=%.3g)', res(ic).c);
    end

    if strcmp(mode, 'smallest')
        set(ax, 'XDir', 'reverse');
        xlabel(ax, sprintf('index i (%d \\rightarrow 1)', k), 'FontSize', 12);
        title(ax, sprintf('smallest %d eigenvalues', k), 'FontSize', 12);
    else
        set(ax, 'XDir', 'normal');
        xlabel(ax, sprintf('index i (1 \\rightarrow %d)', k), 'FontSize', 12);
        title(ax, sprintf('largest %d eigenvalues', k), 'FontSize', 12);
    end
    set(ax, 'YScale', 'log', 'FontSize', 12, 'Box', 'on');
    ylabel(ax, '\lambda_i  (log scale)', 'FontSize', 12);
    legend(ax, legH, legL, 'Location', 'best', 'FontSize', 9, 'Box', 'on', ...
           'NumColumns', 2);
    hold(ax, 'off');
end

function t = ramp_t(ic, nC)
%RAMP_T  Light->dark interpolation parameter in [0,1] for color index ic.
    if nC == 1
        t = 1;
    else
        t = (ic - 1) / (nC - 1);
    end
end

function s = cond_summary(res)
%COND_SUMMARY  One-line tex string of cond(A) and cond(M) per c.
    parts = cell(1, numel(res));
    for ic = 1:numel(res)
        parts{ic} = sprintf('c=%.3g: \\kappa(A)=%.2e, \\kappa(M)=%.2e', ...
                            res(ic).c, res(ic).condA, res(ic).condM);
    end
    s = strjoin(parts, '    |    ');
end
