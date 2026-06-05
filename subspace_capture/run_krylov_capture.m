% RUN_KRYLOV_CAPTURE  Does the (P)CG Krylov subspace capture the small eigenspace?
%
% Question: for the right-hand side b the solver actually uses, does the Krylov
% subspace built up during CG capture the smallest eigenvectors of the operator
% it solves?  If it does, those Ritz directions could be recycled for deflation
% "for free".  The study is run on BOTH systems the solver might use:
%   Z = A     : unconditioned CG on A x = b.  Krylov space K_j(A, b) vs the
%               smallest eigenvectors of A.
%   Z = Tsym  : ICHOL-preconditioned PCG, i.e. CG on Tsym y = b_hat with
%               Tsym = L^{-1} A L^{-T}.  Krylov space K_j(Tsym, b_hat) vs the
%               smallest eigenvectors of Tsym.
%
% Setup (matches subspace_capture/run_inverse_subspace_iter.m):
%   A = D_II + dt*K_II  (sphere snapshot, latitude banding C=60, t=0),
%   L = ichol(A,'nofill'),  Tsym = L^{-1} A L^{-T}.
%   V_true = smallest k eigenvectors of Z (read from the existing caches).
%
% Right-hand side = the EXACT solver RHS at the first step.  On the closed
% sphere (no boundary) with zero initial condition the solver's rhsI reduces to
% the mass-weighted KL noise term (solve_deflate_M_P.m:166-185):
%       b = sigma*sqrt(dt) * D_II * (Phi * z),   z ~ N(0, I_Kmodes),
% where Phi = eval_cosine_modes(p, kvec, bbox), kvec = generate_kvec(Kmodes).
% (P)CG normalizes the RHS and the Krylov subspace is scale-invariant, so sigma
% and sqrt(dt) are irrelevant -- only the DIRECTION D_II*(Phi*z) matters, i.e.
% only Kmodes and the noise seed.
%
% For Z = A, CG on A x = b with x0 = 0 has r0 = b, so the Krylov space is
% K_j(A, b).  For Z = Tsym, PCG on A x = b with M = L L^T is equivalent to CG on
% Tsym y = b_hat with b_hat = L^{-1} b (x0 = 0 => r0 = b_hat), so the Krylov
% space is K_j(Tsym, b_hat).  In each case the orthonormal Lanczos basis
% Q(:,1:j) spans that space; we measure how well Q(:,1:j) captures V_true as j
% grows, up to the (P)CG convergence dimension J.
%
% Because a single-RHS Krylov space of dimension j captures at most ~j vectors,
% capture is reported for several target sizes k in {10, 50, 500}: the smallest
% modes (small k) are captured early, the full 500 lag.
%
% Outputs (subspace_capture/output_krylov_capture/):
%   results.mat / results.csv
%   capture_fraction_A.pdf      — capture fraction (res<1%) vs Krylov dim j, Z=A
%   mean_residual_A.pdf         — mean residual over the target set vs j, Z=A
%   capture_fraction_Tsym.pdf   — same, Z = L^{-1} A L^{-T}
%   mean_residual_Tsym.pdf
%   aggregate_2x2.png           — all four panels (rows = Z, cols = metric)
%
% Usage:
%   cd subspace_capture
%   run_krylov_capture

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);
addpath(thisFileDir);

outDir   = fullfile(thisFileDir, 'output_krylov_capture');
cacheDir = fullfile(thisFileDir, 'output', 'cache');   % REUSE existing eig cache
if ~isfolder(outDir),   mkdir(outDir);   end
if ~isfolder(cacheDir), mkdir(cacheDir); end

%% --- Snapshot configuration (matches the other drivers / the cache key) ----
h0          = 0.05;
contrast    = 60;
t_snap      = 0;
dt          = 1;
Tstep       = 100;
Tmax        = Tstep * dt;
mesh_method = 'pdetoolbox';

%% --- Experiment configuration ---------------------------------------------
k          = 500;                 % cached ground-truth eigenvectors of Tsym
ktars      = [10, 50, 500];       % "small eigenspace" target sizes to report
Kmodes     = 50;                  % # KL cosine modes in the solver RHS
sigma      = 1;                   % RHS scale (irrelevant to the Krylov subspace)
seed       = 1;                   % noise-realization seed
bbox       = [-1 1 -1 1];         % unit-sphere x,y extent for the cosine modes
tol        = 1e-8;                % PCG convergence tolerance (SOLVER_TOL)
maxit      = 400;                 % PCG / Lanczos cap

%% --- Build baseline snapshot A + L ----------------------------------------
fprintf('\n--- Building sphere snapshot (h0=%.4g, %s) ---\n', h0, mesh_method);
[A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
n  = msh.numIN;
Lt = L';
fprintf('A: %d x %d, nnz=%d, numIN=%d, numB=%d\n', ...
        size(A,1), size(A,2), nnz(A), msh.numIN, msh.numB);

%% --- Ground-truth small eigenpairs of A and Tsym (existing caches) ---------
[V_true_A, ~, ~] = load_or_compute_eigs_A(cacheDir, A, k);
[V_true_T, ~, ~] = load_or_compute_eigs_Tsym(cacheDir, A, L, k);
fprintf('V_true_A (smallest %d eigvecs of A)   : %d x %d\n', ...
        k, size(V_true_A,1), size(V_true_A,2));
fprintf('V_true_T (smallest %d eigvecs of Tsym): %d x %d\n', ...
        k, size(V_true_T,1), size(V_true_T,2));

%% --- Exact solver RHS b at step 1 (zero IC) = mass-weighted KL noise -------
kvec = generate_kvec(Kmodes);
Phi  = src.forcing.eval_cosine_modes(msh.p, kvec, bbox);   % numIN x Kmodes
rng(seed);
z    = randn(Kmodes, 1);
b    = sigma * sqrt(dt) * (msh.D_II * (Phi * z));           % exact solver rhsI
fprintf('RHS b: Kmodes=%d, seed=%d, ||b||=%.4e\n', Kmodes, seed, norm(b));

%% --- The two systems studied ----------------------------------------------
% Z = A    : unconditioned CG on A x = b,  r0 = b (x0 = 0),  K_j(A, b).
% Z = Tsym : PCG(A,b,M=LL^T) <=> CG(Tsym, b_hat),  b_hat = L^{-1} b,
%            Tsym = L^{-1} A L^{-T},  r0 = b_hat,  K_j(Tsym, b_hat).
b_hat = L \ b;
systems = struct( ...
    'Z',      {'A',                      'Tsym'}, ...
    'apply',  {@(X) A * X,               @(X) L \ (A * (Lt \ X))}, ...
    'v1',     {b,                        b_hat}, ...
    'V_true', {V_true_A,                 V_true_T}, ...
    'precon', {false,                    true});

%% --- Per-system: RHS energy, (P)CG iterations, Lanczos basis, capture sweep -
rows = [];
J_by_Z = struct('A', NaN, 'Tsym', NaN);
for is = 1:numel(systems)
    sys   = systems(is);
    Zk    = sys.Z;
    v1    = sys.v1;
    Vtrue = sys.V_true;

    fprintf('\n=== Z = %s ===\n', Zk);

    % How much of the starting residual already lives in the small eigenspace?
    v1_unit = v1 / norm(v1);
    for it = 1:numel(ktars)
        kt = ktars(it);
        e  = norm(Vtrue(:,1:kt)' * v1_unit)^2;
        fprintf('energy of r0 in smallest %3d eigvecs of %-4s: %.4f\n', kt, Zk, e);
    end

    % (P)CG convergence dimension J (the actual iteration count).
    if sys.precon
        [~, fl, rr, J] = pcg(A, b, tol, maxit, L, Lt);
        fprintf('ICHOL-PCG: flag=%d, relres=%.2e, converged in J=%d iterations\n', ...
                fl, rr, J);
    else
        [~, fl, rr, J] = pcg(A, b, tol, maxit);
        fprintf('plain CG : flag=%d, relres=%.2e, converged in J=%d iterations\n', ...
                fl, rr, J);
    end
    J_by_Z.(Zk) = J;

    % Lanczos basis of the Krylov space with full reorthogonalization.
    Jdim = min(maxit, J + 5);      % build a few past convergence for context
    Q    = build_lanczos_basis(sys.apply, v1, Jdim);
    Jdim = size(Q, 2);             % may be < Jdim on invariant-subspace breakdown
    fprintf('Built Krylov/Lanczos basis: %d vectors\n', Jdim);

    % Capture sweep over Krylov dimension j.
    for it = 1:numel(ktars)
        kt = ktars(it);
        Vk = Vtrue(:, 1:kt);
        for j = 1:Jdim
            cap = src.precond.subspace_capture(Vk, Q(:, 1:j));
            info = struct( ...
                'Z',                 Zk, ...
                'ktar', kt, 'jdim', j, ...
                'capture_frac_1pct', cap.n_res_below_1pct / kt, ...
                'n_res_below_1pct',  cap.n_res_below_1pct, ...
                'mean_residual',     cap.mean_residual, ...
                'max_residual',      cap.max_residual);
            rows = [rows; info];                                      %#ok<AGROW>
        end
        fprintf('  ktar=%3d : at J=%d capture_frac=%.3f (count=%d), mean_res=%.3e\n', ...
                kt, J, rows(end).capture_frac_1pct, rows(end).n_res_below_1pct, ...
                rows(end).mean_residual);
    end
end

%% --- Save -----------------------------------------------------------------
meta = struct('n', n, 'k', k, 'ktars', ktars, 'Kmodes', Kmodes, 'seed', seed, ...
              'tol', tol, 'maxit', maxit, 'J_A', J_by_Z.A, 'J_Tsym', J_by_Z.Tsym, ...
              'contrast', contrast, 'h0', h0, 't_snap', t_snap);
save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');
write_results_csv(fullfile(outDir, 'results.csv'), rows);
fprintf('\nresults.mat and results.csv written to:\n  %s\n', outDir);

%% --- Plots ----------------------------------------------------------------
fprintf('\n--- Rendering plots ---\n');
make_krylov_plots(rows, ktars, J_by_Z, outDir);

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

function [V_true, lam_cut, lam_first] = load_or_compute_eigs_A(cacheDir, A, k)
%LOAD_OR_COMPUTE_EIGS_A  Smallest k+1 eigenpairs of A; return first k vecs,
%   the (k+1)-th eigenvalue as the cutoff stop-band edge (lam_cut), and the
%   smallest eigenvalue (lam_first = D(1), the left spectral end).
    cachePath = fullfile(cacheDir, sprintf('eigsA_k%d.mat', k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_true    = S.V(:, 1:k);
        lam_cut   = S.D(k + 1);
        lam_first = S.D(1);
        fprintf('Loaded cached %s\n', cachePath);
        return;
    end
    fprintf('Computing smallest %d eigenpairs of A (one-time)...\n', k + 1);
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    [Vraw, Dmat] = eigs(A, k + 1, 'smallestabs', opts);
    [D, idx]     = sort(real(diag(Dmat)), 'ascend');
    V            = real(Vraw(:, idx));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_true    = V(:, 1:k);
    lam_cut   = D(k + 1);
    lam_first = D(1);
end

function [V_true, lam_cut, lam_first] = load_or_compute_eigs_Tsym(cacheDir, A, L, k)
%LOAD_OR_COMPUTE_EIGS_TSYM  Smallest k+1 eigenpairs of T_sym = L^{-1}AL^{-T}.
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
    [Vraw, Dmat] = eigs(Tinv, size(A, 1), k + 1, 'smallestabs', opts);
    [D, idx]     = sort(real(diag(Dmat)), 'ascend');
    V            = real(Vraw(:, idx));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_true    = V(:, 1:k);
    lam_cut   = D(k + 1);
    lam_first = D(1);
end

function Q = build_lanczos_basis(Aapply, v1, m)
%BUILD_LANCZOS_BASIS  Orthonormal basis of the Krylov space K_m(A, v1).
%   Lanczos with full (twice) reorthogonalization for a numerically clean
%   basis -- this is the exact-arithmetic Krylov subspace that CG/PCG spans.
%   Returns Q with up to m columns (fewer on invariant-subspace breakdown).
    n   = numel(v1);
    Q   = zeros(n, m);
    q   = v1 / norm(v1);
    Q(:, 1) = q;
    jj  = 1;
    for j = 1:m-1
        w = Aapply(Q(:, j));
        % Full reorthogonalization against all previous basis vectors (twice).
        w = w - Q(:, 1:j) * (Q(:, 1:j)' * w);
        w = w - Q(:, 1:j) * (Q(:, 1:j)' * w);
        beta = norm(w);
        if beta < 1e-12 * sqrt(n)
            break;   % invariant subspace reached
        end
        Q(:, j+1) = w / beta;
        jj = j + 1;
    end
    Q = Q(:, 1:jj);
end

function write_results_csv(csvPath, rows)
%WRITE_RESULTS_CSV  Flat CSV of the capture sweep.
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['Z,ktar,jdim,capture_frac_1pct,n_res_below_1pct,', ...
                  'mean_residual,max_residual\n']);
    for i = 1:numel(rows)
        r = rows(i);
        fprintf(fid, '%s,%d,%d,%g,%g,%g,%g\n', ...
                r.Z, r.ktar, r.jdim, r.capture_frac_1pct, r.n_res_below_1pct, ...
                r.mean_residual, r.max_residual);
    end
    fclose(fid);
end

function f = make_latitude_banding_contrast(Tmax, contrast)
%MAKE_LATITUDE_BANDING_CONTRAST  Latitude-banding kappa (kept LOCAL).
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
    theta = acos(max(min(z, 1), -1));
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

function kvec = generate_kvec(K)
%GENERATE_KVEC  K wavenumber pairs (kx,ky) sorted by ascending frequency.
%   Copied verbatim from solve_deflate_M_P.m so the RHS matches the solver.
    candidates = [];
    maxk = ceil(sqrt(2*K)) + 1;
    for kx = 0:maxk
        for ky = 0:maxk
            if kx == 0 && ky == 0, continue; end
            candidates = [candidates; kx ky kx^2+ky^2]; %#ok<AGROW>
        end
    end
    [~, idx] = sortrows(candidates, [3 1]);
    kvec = candidates(idx(1:K), 1:2);
end

%% =========================================================================
%% Rendering
%% =========================================================================
function specs = krylov_panel_specs()
%KRYLOV_PANEL_SPECS  The four (metric, Z) capture panels, shared by PDF + aggregate.
%   Field `tag` is the file stem for the per-panel PDFs.
    specs = struct( ...
        'metric', {'capture_frac_1pct',           'mean_residual', ...
                   'capture_frac_1pct',           'mean_residual'}, ...
        'Z',      {'A',                            'A', ...
                   'Tsym',                         'Tsym'}, ...
        'Zlabel', {'A',                            'A', ...
                   'L^{-1} A L^{-T}',              'L^{-1} A L^{-T}'}, ...
        'yscale', {'linear',                       'log', ...
                   'linear',                       'log'}, ...
        'ylabel', {'capture fraction (res < 1%)',  'mean residual over target set', ...
                   'capture fraction (res < 1%)',  'mean residual over target set'}, ...
        'titlebit', {'Krylov capture fraction',    'Krylov mean residual', ...
                     'Krylov capture fraction',    'Krylov mean residual'}, ...
        'legloc', {'northwest',                    'northeast', ...
                   'northwest',                    'northeast'}, ...
        'tag',    {'capture_fraction_A',           'mean_residual_A', ...
                   'capture_fraction_Tsym',        'mean_residual_Tsym'});
end

function make_krylov_plots(rows, ktars, J_by_Z, out_dir)
%MAKE_KRYLOV_PLOTS  Four per-panel PDFs + one 2x2 aggregate PNG.
%   Rows of the 2x2 grid = operator Z {A, Tsym}; columns = metric {capture
%   fraction, mean residual}.  J_by_Z is a struct with fields A and Tsym giving
%   the (P)CG convergence dimension marked on each panel.
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    specs = krylov_panel_specs();
    for ip = 1:numel(specs)
        fig = figure('Visible', 'off', 'Units', 'inches', ...
                     'Position', [0 0 5.6 3.4], 'Color', 'w');
        draw_krylov_panel(axes(fig), rows, ktars, J_by_Z.(specs(ip).Z), specs(ip));
        outfile = fullfile(out_dir, [specs(ip).tag '.pdf']);
        exportgraphics(fig, outfile, 'ContentType', 'vector');
        close(fig);
        fprintf('Wrote %s\n', outfile);
    end

    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 11.4 7.2], 'Color', 'w');
    tl = tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for ip = 1:numel(specs)
        draw_krylov_panel(nexttile(tl), rows, ktars, J_by_Z.(specs(ip).Z), specs(ip));
    end
    title(tl, 'Does the (P)CG Krylov subspace capture the small eigenspace? (exact solver RHS)', ...
          'FontWeight', 'bold', 'FontSize', 12);
    outfile = fullfile(out_dir, 'aggregate_2x2.png');
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
    fprintf('Wrote %s\n', outfile);
end

function draw_krylov_panel(ax, rows, ktars, J, spec)
%DRAW_KRYLOV_PANEL  One (metric, Z) capture panel, one curve per target size.
    colors = [0.20 0.40 0.80;    % k small  (blue)
              0.85 0.45 0.10;    % k mid    (orange)
              0.00 0.00 0.00];   % k large  (black)
    marks  = {'o', 's', '^'};

    hold(ax, 'on');
    handles = gobjects(1, numel(ktars));
    legs    = cell(1, numel(ktars));
    yall    = [];
    for it = 1:numel(ktars)
        kt   = ktars(it);
        mask = [rows.ktar] == kt & strcmp({rows.Z}, spec.Z);
        sub  = rows(mask);
        [xs, ord] = sort([sub.jdim]);
        ys        = [sub.(spec.metric)];
        ys        = ys(ord);
        ci = mod(it - 1, size(colors, 1)) + 1;
        handles(it) = plot(ax, xs, ys, ['-' marks{min(it,numel(marks))}], ...
                           'Color', colors(ci,:), 'MarkerFaceColor', colors(ci,:), ...
                           'MarkerSize', 3.5, 'LineWidth', 1.3, ...
                           'MarkerIndices', 1:max(1,round(numel(xs)/25)):numel(xs));
        legs{it} = sprintf('smallest %d', kt);
        yall = [yall, ys(isfinite(ys))]; %#ok<AGROW>
    end

    % (P)CG convergence marker.
    if strcmp(spec.Z, 'Tsym'), solver = 'PCG'; else, solver = 'CG'; end
    yl = ylim(ax);
    hJ = xline(ax, J, '--', sprintf('%s converged (J=%d)', solver, J), ...
               'Color', [0.4 0.4 0.4], 'LineWidth', 1.0, ...
               'LabelVerticalAlignment', 'bottom', ...
               'LabelHorizontalAlignment', 'left', 'FontSize', 7); %#ok<NASGU>
    ylim(ax, yl);
    hold(ax, 'off');

    set(ax, 'XScale', 'linear', 'YScale', spec.yscale, 'Box', 'on', ...
            'LineWidth', 0.6, 'FontSize', 9);
    if strcmp(spec.yscale, 'linear')
        ylim(ax, [-0.02, 1.05]);
        set(ax, 'YTick', 0:0.2:1);
    end
    xlabel(ax, sprintf('Krylov dimension j  (= %s iteration)', solver), ...
           'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, spec.ylabel, 'FontSize', 10, 'FontWeight', 'bold');
    title(ax, sprintf('%s  (Z = %s)', spec.titlebit, spec.Zlabel), ...
          'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');

    lgd = legend(ax, handles, legs, 'Location', spec.legloc, 'Box', 'on', ...
                 'EdgeColor', [0.65 0.65 0.65], 'Color', 'white', ...
                 'FontSize', 8, 'Interpreter', 'none');
    lgd.ItemTokenSize = [16, 6];
end
