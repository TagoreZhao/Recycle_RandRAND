% RUN_INVERSE_SUBSPACE_ITER  Exact-inverse subspace-iteration capture study.
%
% Studies how PLAIN subspace iteration driven by the EXACT INVERSE of the
% preconditioned operator captures the smallest eigenvectors of that operator,
% and how the captured subspace evolves as the number of iterations grows.
%
% Iteration operator (exact inverse of the symmetric preconditioned operator):
%   Z = Tsym^{-1} = L^T A^{-1} L,   Tsym = L^{-1} A L^{-T},  L = ichol(A,'nofill').
% Plain power iteration with this Z drives a starting block toward the dominant
% eigenvectors of Tsym^{-1} == the SMALLEST eigenvectors of Tsym.  We use
% src.precond.subspace_iter_plain (NO re-orthogonalization inside the loop) and
% orthonormalize only ONCE at the end, just before measuring capture.
%
% Four starting blocks are compared in two matched-width ablation pairs, all
% built from the preconditioned operator T:
%   width m     : gaussian        = randn(n,m)         vs  sketched_tent_T = Pt_T*randn(nc_T,m)
%   width nc_T  : gaussian_tent   = randn(n,nc_T)      vs  tent_T          = full(Pt_T)
% (nc_T >= m is the raw tentative width.)  This isolates "tentative vs Gaussian"
% at each width.
%
% Two POLYNOMIAL-FILTER families are overlaid on the same two panels (only for
% the width-m blocks gaussian / sketched_tent_T), so all methods share one
% x-axis q = number of operator applications:
%   chebyshev : src.precond.chebyshev_apply  -- high-pass T_q on forward Tsym (q matvecs)
%   power_iz  : src.precond.min_subspace_iter -- (I - Tsym/lam_max)^q          (q matvecs)
% For the inverse method q is the iteration power (q exact Tsym^{-1} solves);
% for the polynomial filters q is the degree.  This contrasts the fast-but-
% unstable exact inverse against the slower-but-stable polynomial filters.
%
% Outputs (subspace_capture/output_inverse_iter/):
%   results.mat / results.csv
%   max_residual_inv.pdf       — max per-vector residual (log y) vs iteration q
%   capture_fraction_inv.pdf   — fraction of vectors with res<1% vs iteration q
%   aggregate_1x2.png          — both panels side by side (convenience)
%
% Ground truth (smallest k eigenvectors of Tsym) is read from the cache that
% run_subspace_capture.m already populated (output/cache/eigsTsym_k500.mat); no
% eigendecomposition is recomputed if that cache is present.
%
% Usage:
%   cd subspace_capture
%   run_inverse_subspace_iter

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);     % so `src.*` packages resolve
addpath(thisFileDir);  % local helpers

outDir   = fullfile(thisFileDir, 'output_inverse_iter');
cacheDir = fullfile(thisFileDir, 'output', 'cache');  % REUSE existing eig cache
if ~isfolder(outDir),   mkdir(outDir);   end
if ~isfolder(cacheDir), mkdir(cacheDir); end

%% --- Snapshot configuration (matches run_subspace_capture so the cache key fits) ---
h0          = 0.05;     % baseline mesh edge length
contrast    = 60;       % kappa_max / kappa_min
t_snap      = 0;        % snapshot time level
dt          = 1;
Tstep       = 100;
Tmax        = Tstep * dt;     % = 100
mesh_method = 'pdetoolbox';

%% --- Sweep configuration --------------------------------------------------
k            = 500;
m            = 2 * k;
iters        = [0 1 2 3 4 5 6 8 10 12 16 20];   % x-axis: # subspace iterations
theta        = 0.05;
maxAggSize   = 16;
seed         = 1;
drop_rel_tol = 1e-8;

P0_kinds = {'gaussian', 'sketched_tent_T', 'tent_T', 'gaussian_tent'};

% Methods sharing the common x-axis q (= operator applications):
%   inverse   : plain subspace iteration with Z = Tsym^{-1} (q exact solves)
%   chebyshev : Chebyshev high-pass filter on forward Tsym  (q matvecs)
%   power_iz  : damped power filter (I - Tsym/lam_max)^q     (q matvecs)
% The polynomial filters are overlaid only on the two width-m blocks.
poly_blocks = {'gaussian', 'sketched_tent_T'};

%% --- Build baseline snapshot A + L ----------------------------------------
fprintf('\n--- Building sphere snapshot (h0=%.4g, %s) ---\n', h0, mesh_method);
[A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
n  = msh.numIN;
Lt = L';
fprintf('A: %d x %d, nnz=%d, sym=%d   nnz(L)=%d\n', ...
        size(A,1), size(A,2), nnz(A), issymmetric(A), nnz(L));

%% --- Ground-truth small eigenpairs of Tsym (read from existing cache) ------
[V_true_T, lam_cut, ~] = load_or_compute_eigs_Tsym(cacheDir, A, L, k);
fprintf('V_true (smallest %d eigvecs of Tsym): %d x %d  (lam_cut=%.4e)\n', ...
        k, size(V_true_T,1), size(V_true_T,2), lam_cut);

%% --- Operators -------------------------------------------------------------
% Exact inverse  Z = Tsym^{-1} = L^T A^{-1} L  (for the 'inverse' method).
dA        = decomposition(A, 'chol');
invApply  = @(X) Lt * (dA \ (L * X));
% Forward Tsym = L^{-1} A L^{-T}  (for the polynomial-filter methods).
Zfun_Tsym = @(X) L \ (A * (Lt \ X));
% Top of the Tsym spectrum (reused from cache) -- needed by the poly filters.
lam_max_T = load_or_compute_lam_max(cacheDir, 'Tsym', Zfun_Tsym, n);
fprintf('lam_max(Tsym) = %.4e\n', lam_max_T);

%% --- Tentative prolongator from the preconditioned operator T --------------
T_sparse = build_Tsym_sparse(L, Lt, A, drop_rel_tol);
fprintf(['Tsym sparsification: nnz=%d (%.2f%% of n^2), ', ...
         'max|T|=%.3e\n'], ...
        nnz(T_sparse), 100 * nnz(T_sparse) / n^2, ...
        max(abs(nonzeros(T_sparse))));
[Pt_T, nc_T] = build_tent_at_least(T_sparse, theta, maxAggSize, m);
fprintf('Pt_T: nc=%d  (m=%d, maxAggSize=%d)\n', nc_T, m, maxAggSize);

%% --- Pre-build all four P0 blocks -----------------------------------------
P0_cache = struct();
for ik = 1:numel(P0_kinds)
    kind = P0_kinds{ik};
    rng(seed);
    [P0, ncols] = build_P0(kind, Pt_T, nc_T, n, m);
    P0_cache.(kind).P  = P0;
    P0_cache.(kind).nc = ncols;
    fprintf('P0 %-16s: %d x %d\n', kind, size(P0,1), ncols);
end

%% --- Sweep ----------------------------------------------------------------
% Operator handles + spectral bounds shared by the per-q runs.
ops = struct('invApply', invApply, 'Zfun_Tsym', Zfun_Tsym, ...
             'lam_cut', lam_cut, 'lam_max', lam_max_T, 'n', n);

rows = [];
for ik = 1:numel(P0_kinds)
    kind  = P0_kinds{ik};
    P0    = P0_cache.(kind).P;
    ncols = P0_cache.(kind).nc;

    % Inverse iteration for every block; polynomial filters only on width-m blocks.
    if any(strcmp(kind, poly_blocks))
        methods = {'inverse', 'chebyshev', 'power_iz'};
    else
        methods = {'inverse'};
    end

    for im = 1:numel(methods)
        method = methods{im};
        for iq = 1:numel(iters)
            q    = iters(iq);
            info = run_one_method(method, ops, P0, q, V_true_T);
            info.P0_kind  = kind;
            info.P0_ncols = ncols;
            info.method   = method;
            info.iter     = q;
            rows = [rows; info];                                      %#ok<AGROW>
            fprintf('  %-16s %-10s q=%2d : max_res=%.3e  capture=%.3f\n', ...
                    kind, method, q, info.max_residual, info.capture_frac_1pct);
        end
    end
end

%% --- Save -----------------------------------------------------------------
meta = struct('n', n, 'k', k, 'm', m, 'iters', iters, ...
              'contrast', contrast, 'h0', h0, 't_snap', t_snap, ...
              'nc_T', nc_T, 'Z', 'Tsym_inverse', ...
              'lam_cut', lam_cut, 'lam_max_T', lam_max_T);
save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');
write_results_csv(fullfile(outDir, 'results.csv'), rows);
fprintf('\nresults.mat and results.csv written to:\n  %s\n', outDir);

%% --- Render the two plots --------------------------------------------------
fprintf('\n--- Rendering plots ---\n');
make_inverse_plots(rows, outDir);

fprintf('\nDone.\n');

%% =========================================================================
%% Local helpers
%% =========================================================================
function [A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method)
%BUILD_SNAPSHOT  Mesh + A = D_II + dt*K_II + L = ichol(A,'nofill').
    msh      = src.discretization.build_sphere_mesh(h0, false, mesh_method);
    kappaFun = make_latitude_banding_contrast(Tmax, contrast);
    A        = assemble_snapshot_A(msh, kappaFun, dt, t_snap);
    L        = ichol_with_fallback(A);
end

function A = assemble_snapshot_A(msh, kappaFun, dt, tcur)
%ASSEMBLE_SNAPSHOT_A  Build A = D_II + dt*K_II at one time level (closed sphere).
    numIN   = msh.numIN;
    kappa_e = kappaFun(msh.cent(:,1), msh.cent(:,2), msh.cent(:,3), tcur);
    Vscale  = msh.Vunit .* repelem(kappa_e, 9);
    Vii     = Vscale(msh.idxII);
    K_II    = sparse(msh.I_II, msh.J_II, Vii, numIN, numIN);
    A       = msh.D_II + dt * K_II;
    A       = 0.5 * (A + A.');
end

function L = ichol_with_fallback(A)
%ICHOL_WITH_FALLBACK  ichol(A,'nofill') with diagcomp safety net.
    try
        L = ichol(A, struct('type', 'nofill'));
    catch
        alpha    = max(sum(abs(A), 2) ./ diag(A)) - 2;
        diagcomp = max(alpha, 0);
        L = ichol(A, struct('type', 'nofill', 'diagcomp', diagcomp));
    end
end

function [V_true, lam_cut, lam_first] = load_or_compute_eigs_Tsym(cacheDir, A, L, k)
%LOAD_OR_COMPUTE_EIGS_TSYM  Smallest k+1 eigenpairs of T_sym = L^{-1}AL^{-T}.
%   Returns first k vecs, the cutoff D(k+1), and the smallest eigenvalue D(1).
    cachePath = fullfile(cacheDir, sprintf('eigsTsym_k%d.mat', k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_true    = S.V(:, 1:k);
        lam_cut   = S.D(k + 1);
        lam_first = S.D(1);
        fprintf('Loaded cached %s\n', cachePath);
        return;
    end
    fprintf('Computing smallest %d eigenpairs of L^{-1} A L^{-T} (one-time)...\n', k + 1);
    Lt   = L';
    dA   = decomposition(A, 'chol');
    Tinv = @(x) Lt * (dA \ (L * x));
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    t0   = tic;
    [Vraw, Dmat] = eigs(Tinv, size(A, 1), k + 1, 'smallestabs', opts);
    [D, idx]     = sort(real(diag(Dmat)), 'ascend');
    V            = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_true    = V(:, 1:k);
    lam_cut   = D(k + 1);
    lam_first = D(1);
end

function T_sparse = build_Tsym_sparse(L, Lt, A, drop_rel_tol)
%BUILD_TSYM_SPARSE  Materialize Tsym = L^{-1} A L^{-T}, symmetrize, drop near-zeros.
    Tdense = (L \ full(A)) / Lt;
    Tdense = 0.5 * (Tdense + Tdense.');
    cutoff = drop_rel_tol * max(abs(Tdense(:)));
    Tdense(abs(Tdense) < cutoff) = 0;
    T_sparse = sparse(Tdense);
end

function [Pt, nc] = build_tent_at_least(M, theta, maxAggSize, m_min)
%BUILD_TENT_AT_LEAST  Tentative prolongator Pt with nc >= m_min.
    lo = 1;  hi = maxAggSize;
    [Pt_hi, nc_hi] = src.precond.tentative_prolongator(M, theta, hi);
    if nc_hi >= m_min
        Pt = Pt_hi;  nc = nc_hi;  return;
    end
    [Pt_lo, nc_lo] = src.precond.tentative_prolongator(M, theta, lo);
    if nc_lo < m_min
        error('build_tent_at_least:tooSmallN', ...
              'cap=1 yields nc=%d < m_min=%d (n is too small).', ...
              nc_lo, m_min);
    end
    while hi - lo > 1
        mid = floor((lo + hi) / 2);
        [Pt_mid, nc_mid] = src.precond.tentative_prolongator(M, theta, mid);
        if nc_mid >= m_min
            lo = mid;  Pt_lo = Pt_mid;  nc_lo = nc_mid;
        else
            hi = mid;  Pt_hi = Pt_mid;  nc_hi = nc_mid;                %#ok<NASGU>
        end
    end
    Pt = Pt_lo;  nc = nc_lo;
end

function [P0, ncols] = build_P0(kind, Pt_T, nc_T, n, m)
%BUILD_P0  Construct one of the four starting blocks, all from the T tentative.
    switch kind
        case 'gaussian'                 % width m
            P0    = randn(n, m);
            ncols = m;
        case 'sketched_tent_T'          % width m
            G     = randn(nc_T, m);
            P0    = Pt_T * G;
            ncols = m;
        case 'tent_T'                   % width nc_T (raw tentative)
            P0    = full(Pt_T);
            ncols = nc_T;
        case 'gaussian_tent'            % width nc_T (Gaussian matched to raw tentative)
            P0    = randn(n, nc_T);
            ncols = nc_T;
        otherwise
            error('build_P0: unknown kind %s', kind);
    end
end

function info = run_one_method(method, ops, P0, q, V_true)
%RUN_ONE_METHOD  Apply one method at "work level" q, then measure capture.
%   q counts operator applications for all methods:
%     inverse   : q plain applications of Z = Tsym^{-1} (no reorth), orth at end.
%     chebyshev : degree-q Chebyshev high-pass filter on forward Tsym, orth at end.
%     power_iz  : degree-q damped power filter (I - Tsym/lam_max)^q (final orth
%                 done inside min_subspace_iter).
%   At q=0 all three reduce to orth(P0) -- a built-in consistency check.
    info = new_capture_info();
    t0   = tic;
    try
        switch method
            case 'inverse'
                Y = src.precond.subspace_iter_plain(ops.invApply, P0, q);
                Q = orth(Y);                   % only orth -- at the very end
            case 'chebyshev'
                Y = src.precond.chebyshev_apply(ops.Zfun_Tsym, P0, q, ...
                                                ops.lam_cut, ops.lam_max);
                Q = orth(Y);
            case 'power_iz'
                Dinv = (1 / ops.lam_max) * ones(ops.n, 1);
                Q = src.precond.min_subspace_iter(ops.Zfun_Tsym, P0, q, ...
                                                  Dinv, 1.0, false);
            otherwise
                error('run_one_method: unknown method %s', method);
        end
        info = fill_capture_info(info, V_true, Q);
    catch ME
        info.err = regexprep(ME.message, '\n.*', '');
        warning('run_inverse_subspace_iter:run_one_method_failed', ...
                'method=%s q=%d failed: %s', method, q, info.err);
    end
    info.time_seconds = toc(t0);
end

function lam_max = load_or_compute_lam_max(cacheDir, kind, op, n)
%LOAD_OR_COMPUTE_LAM_MAX  Return cached lam_max(Z); compute and cache if missing.
    cachePath = fullfile(cacheDir, sprintf('lam_max_%s.mat', kind));
    if isfile(cachePath)
        S = load(cachePath, 'lam_max');
        lam_max = S.lam_max;
        fprintf('Loaded cached %s\n', cachePath);
        return;
    end
    fprintf('Computing lam_max(%s) ...\n', kind);
    opts = struct('Tolerance', 1e-8, 'MaxIterations', 5000, ...
                  'IsFunctionSymmetric', true);
    if isnumeric(op)
        d = eigs(op, 1, 'largestabs', opts);
    else
        d = eigs(op, n, 1, 'largestabs', opts);
    end
    lam_max = real(d);
    save(cachePath, 'lam_max', '-v7');
end

function info = new_capture_info()
%NEW_CAPTURE_INFO  Empty result struct.
    info = struct('max_residual', NaN, 'mean_residual', NaN, ...
                  'n_res_below_1pct', NaN, 'n_res_below_0p1pct', NaN, ...
                  'capture_frac_1pct', NaN, 'max_principal_angle', NaN, ...
                  'time_seconds', NaN, 'ok', false, 'err', '');
end

function info = fill_capture_info(info, V_true, Q)
%FILL_CAPTURE_INFO  Populate residual/capture metrics from a computed basis Q.
    cap = src.precond.subspace_capture(V_true, Q);
    info.max_residual        = cap.max_residual;
    info.mean_residual       = cap.mean_residual;
    info.n_res_below_1pct    = cap.n_res_below_1pct;
    info.n_res_below_0p1pct  = cap.n_res_below_0p1pct;
    info.capture_frac_1pct   = cap.n_res_below_1pct / size(V_true, 2);
    info.max_principal_angle = max(cap.principal_angles);
    info.ok = true;
end

function write_results_csv(csvPath, rows)
%WRITE_RESULTS_CSV  Flat CSV of the sweep results.
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['P0_kind,method,P0_ncols,iter,', ...
                  'max_residual,mean_residual,n_res_below_1pct,', ...
                  'n_res_below_0p1pct,capture_frac_1pct,max_principal_angle,', ...
                  'time_seconds\n']);
    for i = 1:numel(rows)
        r = rows(i);
        fprintf(fid, '%s,%s,%d,%d,%g,%g,%g,%g,%g,%g,%g\n', ...
                r.P0_kind, r.method, r.P0_ncols, r.iter, ...
                r.max_residual, r.mean_residual, ...
                r.n_res_below_1pct, r.n_res_below_0p1pct, ...
                r.capture_frac_1pct, r.max_principal_angle, ...
                r.time_seconds);
    end
    fclose(fid);
end

function f = make_latitude_banding_contrast(Tmax, contrast)
%MAKE_LATITUDE_BANDING_CONTRAST  Latitude-banding kappa with adjustable contrast.
%   Kept LOCAL (not in +src): kappa factories are experiment-specific.
    if nargin < 2 || isempty(contrast), contrast = 60; end
    if contrast < 1
        error('make_latitude_banding_contrast:badContrast', ...
              'contrast must be >= 1, got %g', contrast);
    end
    kmin = 1 / sqrt(contrast);
    kmax = sqrt(contrast);
    band_width = 0.25;
    freqs = [sqrt(2), sqrt(3), sqrt(5)];
    f = @(x, y, z, t) latitude_banding_eval(x, y, z, t, Tmax, kmin, kmax, ...
                                            band_width, freqs);
end

function val = latitude_banding_eval(x, y, z, t, Tmax, kmin, kmax, bw, freqs) %#ok<INUSL>
    theta = acos(max(min(z, 1), -1));   % colatitude [0, pi]
    bump = zeros(size(x));
    base_positions = [pi/4, pi/2, 3*pi/4];
    for k = 1:3
        center = base_positions(k) + 0.5 * sin(2 * pi * freqs(k) * t / Tmax);
        center = max(0.1, min(pi - 0.1, center));
        dist = abs(theta - center);
        bump = bump + 0.5 + 0.5 * tanh((bw - dist) / 0.1);
    end
    bump = bump / 3;
    bump = min(bump, 1);
    val = kmin + (kmax - kmin) * bump;
end

%% =========================================================================
%% Rendering: two plots (+ aggregate PNG)
%% =========================================================================
function make_inverse_plots(rows, out_dir)
%MAKE_INVERSE_PLOTS  Two capture plots vs iteration count, four series each.
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    specs = struct( ...
        'metric', {'max_residual',          'capture_frac_1pct'}, ...
        'yscale', {'log',                   'linear'}, ...
        'ylabel', {'max residual',          'capture fraction (res < 1%)'}, ...
        'title',  {'max residual',          'capture fraction'}, ...
        'legloc', {'southwest',             'northeast'}, ...
        'tag',    {'max_residual_inv',      'capture_fraction_inv'});

    for ip = 1:numel(specs)
        fig = figure('Visible', 'off', 'Units', 'inches', ...
                     'Position', [0 0 5.4 3.4], 'Color', 'w');
        draw_inverse_panel(axes(fig), rows, specs(ip));
        outfile = fullfile(out_dir, [specs(ip).tag '.pdf']);
        exportgraphics(fig, outfile, 'ContentType', 'vector');
        close(fig);
        fprintf('Wrote %s\n', outfile);
    end

    % Convenience side-by-side PNG.
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 11 3.6], 'Color', 'w');
    tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for ip = 1:numel(specs)
        draw_inverse_panel(nexttile(tl), rows, specs(ip));
    end
    title(tl, ['Capture vs operator applications q: exact-inverse iteration ', ...
               'vs polynomial filters'], ...
          'FontWeight', 'bold', 'FontSize', 12);
    outfile = fullfile(out_dir, 'aggregate_1x2.png');
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
    fprintf('Wrote %s\n', outfile);
end

function draw_inverse_panel(ax, rows, spec)
%DRAW_INVERSE_PANEL  One (metric vs iteration) panel overlaying the 4 series.
    series = series_spec();
    hold(ax, 'on');
    legs    = cell(1, numel(series));
    handles = gobjects(1, numel(series));
    keep    = false(1, numel(series));
    yall    = [];
    xall    = [];

    for si = 1:numel(series)
        s    = series(si);
        mask = strcmp({rows.P0_kind}, s.P0_kind) & strcmp({rows.method}, s.method);
        sub  = rows(mask);
        if isempty(sub), continue; end
        [xs, ord] = sort([sub.iter]);
        ys        = [sub.(spec.metric)];
        ys        = ys(ord);
        handles(si) = plot(ax, xs, ys, [s.style s.mark], ...
                           'Color', s.color, 'MarkerFaceColor', s.color, ...
                           'MarkerSize', 5, 'LineWidth', 1.2);
        legs{si} = s.label;
        keep(si) = true;
        yall = [yall, ys];  %#ok<AGROW>
        xall = [xall, xs];  %#ok<AGROW>
    end
    hold(ax, 'off');

    set(ax, 'XScale', 'linear', 'YScale', spec.yscale, 'Box', 'on', ...
            'LineWidth', 0.6, 'FontSize', 9);
    if ~isempty(xall)
        xt = unique(xall);
        xlim(ax, [min(xt) - 0.5, max(xt) + 0.5]);
        set(ax, 'XTick', xt);
    end
    if strcmp(spec.yscale, 'linear') && contains(spec.ylabel, 'capture')
        ylim(ax, [-0.02, 1.05]);
        set(ax, 'YTick', 0:0.2:1);
    elseif strcmp(spec.yscale, 'log') && ~isempty(yall)
        yp = yall(yall > 0);
        if ~isempty(yp)
            ylim(ax, [min(yp) * 0.5, max(yp) * 1.5]);
        end
    end

    xlabel(ax, 'number of subspace iterations q', ...
           'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, spec.ylabel, 'FontSize', 10, 'FontWeight', 'bold');
    title(ax, sprintf('%s  (Z = Tsym^{-1})', spec.title), ...
          'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');

    lgd = legend(ax, handles(keep), legs(keep), ...
                 'Location', spec.legloc, 'Box', 'on', ...
                 'EdgeColor', [0.65 0.65 0.65], 'Color', 'white', ...
                 'FontSize', 6.5, 'NumColumns', 2, 'Interpreter', 'none');
    lgd.ItemTokenSize = [14, 6];
end

function series = series_spec()
%SERIES_SPEC  The 8 (block, method) series with consistent visuals.
%   Color encodes the starting block; line-style + marker encode the method.
%   Polynomial filters (chebyshev/power_iz) only exist for the two width-m blocks.
    block_color = struct( ...
        'gaussian',        [0.00 0.00 0.00], ...
        'sketched_tent_T', [0.85 0.33 0.10], ...
        'tent_T',          [0.20 0.70 0.30], ...
        'gaussian_tent',   [0.35 0.35 0.75]);
    block_short = struct( ...
        'gaussian',        'gaussian(m)', ...
        'sketched_tent_T', 'sk_tent_T(m)', ...
        'tent_T',          'tent_T(nc)', ...
        'gaussian_tent',   'gauss_tent(nc)');
    method_style = struct('inverse', '-',  'chebyshev', '--', 'power_iz', ':');
    method_mark  = struct('inverse', 'o',  'chebyshev', 's',  'power_iz', '^');

    % (block, method) pairs in legend order.
    combos = { ...
        'gaussian',        'inverse'; ...
        'gaussian',        'chebyshev'; ...
        'gaussian',        'power_iz'; ...
        'sketched_tent_T', 'inverse'; ...
        'sketched_tent_T', 'chebyshev'; ...
        'sketched_tent_T', 'power_iz'; ...
        'tent_T',          'inverse'; ...
        'gaussian_tent',   'inverse'};

    nS = size(combos, 1);
    series = repmat(struct('P0_kind', '', 'method', '', 'label', '', ...
                           'style', '', 'mark', '', 'color', [0 0 0]), 1, nS);
    for i = 1:nS
        blk = combos{i, 1};  mth = combos{i, 2};
        series(i).P0_kind = blk;
        series(i).method  = mth;
        series(i).label   = sprintf('%s / %s', block_short.(blk), mth);
        series(i).style   = method_style.(mth);
        series(i).mark    = method_mark.(mth);
        series(i).color   = block_color.(blk);
    end
end
