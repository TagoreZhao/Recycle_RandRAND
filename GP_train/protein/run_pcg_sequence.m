function rows = run_pcg_sequence(X, y, c_list, ell0, sigma2, tol, maxit, icholOpts)
%RUN_PCG_SEQUENCE  Solve a swept sequence of SPD kernel-ridge systems with PCG.
%   rows = RUN_PCG_SEQUENCE(X, y, C_LIST, ELL0, SIGMA2, TOL, MAXIT, ICHOLOPTS)
%   builds, for each multiplier c in C_LIST, the SPD system
%
%       A_j = K_{ell_j}(X,X) + SIGMA2 * I,   ell_j = c_j * ELL0,
%
%   and solves A_j x = y twice with MATLAB's built-in PCG: once with no
%   preconditioner and once with an incomplete-Cholesky preconditioner
%   (ichol). Build/solve times, iteration counts, residuals and flags are
%   recorded per system. An ichol failure is caught and recorded; the
%   sequence continues.
%
%   Inputs
%     X         : n-by-d standardized features for this subset.
%     y         : n-by-1 standardized target.
%     c_list    : vector of lengthscale multipliers (one system per entry).
%     ell0      : base lengthscale.
%     sigma2    : ridge regularization (added to the diagonal).
%     tol       : PCG relative-residual tolerance.
%     maxit     : PCG maximum iterations.
%     icholOpts : struct of options passed to ichol (e.g. type, droptol, michol).
%
%   Output
%     rows      : 1-by-numel(c_list) struct array with fields:
%                 n, j, c_j, ell_j,
%                 iters_none, relres_none, time_none, flag_none,
%                 iters_ichol, relres_ichol, time_ichol,
%                 ichol_build_time, flag_ichol, ichol_failed.
%
%   Example
%     rows = run_pcg_sequence(Xn, yn, linspace(0.5,2,20), ell0, 1e-3, 1e-8, 1000, opts);
%
%   Implementation notes
%     - The kernel matrix is dense; ichol is given sparse(A) per the baseline
%       recipe. Memory scales as O(n^2) per system.

    n  = size(X, 1);
    nj = numel(c_list);
    rows = init_rows(n, nj);

    for j = 1:nj
        c_j   = c_list(j);
        ell_j = c_j * ell0;
        A = rbf_kernel_matrix(X, ell_j) + sigma2 * eye(n);

        rows(j).n     = n;
        rows(j).j     = j;
        rows(j).c_j   = c_j;
        rows(j).ell_j = ell_j;

        % --- Solve 1: PCG, no preconditioner ---
        t = tic;
        [~, flag0, relres0, iter0] = pcg(A, y, tol, maxit);
        rows(j).time_none   = toc(t);
        rows(j).iters_none  = iter0;
        rows(j).relres_none = relres0;
        rows(j).flag_none   = flag0;

        % --- Solve 2: PCG + incomplete Cholesky ---
        % ichol on a dense, non-diagonally-dominant kernel often hits a
        % nonpositive pivot; build_ichol_robust escalates diagcomp (a shift
        % applied only to the preconditioner, not to the solved system A).
        try
            [L, alphaUsed, buildT] = build_ichol_robust(A, icholOpts);
            rows(j).ichol_build_time = buildT;
            rows(j).ichol_alpha = alphaUsed;

            ts = tic;
            [~, flag1, relres1, iter1] = pcg(A, y, tol, maxit, L, L');
            rows(j).time_ichol   = toc(ts);
            rows(j).iters_ichol  = iter1;
            rows(j).relres_ichol = relres1;
            rows(j).flag_ichol   = flag1;
            rows(j).ichol_failed = false;
        catch err
            rows(j).ichol_failed = true;
            fprintf(2, '  [n=%d j=%d] ichol failed: %s\n', n, j, err.message);
        end

        fprintf(['  n=%d j=%2d c=%.3f ell=%.4g | none: it=%4d relres=%.1e flag=%d' ...
                 ' | ichol: it=%s relres=%s flag=%s build=%s alpha=%s\n'], ...
                n, j, c_j, ell_j, rows(j).iters_none, rows(j).relres_none, ...
                rows(j).flag_none, ...
                num_or_na(rows(j).iters_ichol), sci_or_na(rows(j).relres_ichol), ...
                num_or_na(rows(j).flag_ichol), sci_or_na(rows(j).ichol_build_time), ...
                num_or_na(rows(j).ichol_alpha));
    end
end

%% --------- local helpers ---------
function rows = init_rows(n, nj)
%INIT_ROWS  Preallocate the result struct array with NaN/false defaults.
    template = struct( ...
        'n', n, 'j', NaN, 'c_j', NaN, 'ell_j', NaN, ...
        'iters_none', NaN, 'relres_none', NaN, 'time_none', NaN, 'flag_none', NaN, ...
        'iters_ichol', NaN, 'relres_ichol', NaN, 'time_ichol', NaN, ...
        'ichol_build_time', NaN, 'ichol_alpha', NaN, ...
        'flag_ichol', NaN, 'ichol_failed', true);
    rows = repmat(template, 1, nj);
end

function s = num_or_na(v)
    if isnan(v), s = 'NA'; else, s = sprintf('%g', v); end
end

function s = sci_or_na(v)
    if isnan(v), s = 'NA'; else, s = sprintf('%.1e', v); end
end
