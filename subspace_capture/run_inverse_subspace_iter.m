% RUN_INVERSE_SUBSPACE_ITER  Exact-inverse subspace-iteration capture study.
%
% Studies how PLAIN subspace iteration driven by the EXACT INVERSE of an SPD
% operator captures that operator's smallest eigenvectors, and how the captured
% subspace evolves as the number of iterations grows.  TWO systems are run with
% identical methodology so they can be compared side by side:
%   Tsym : the split-preconditioned operator
%          Z = Tsym^{-1} = L^T A^{-1} L,  Tsym = L^{-1} A L^{-T},  L = ichol(A,'nofill')
%   A    : the ORIGINAL linear system,  Z = A^{-1}
% Plain power iteration with Z drives a starting block toward the dominant
% eigenvectors of Z == the SMALLEST eigenvectors of the system operator.  We use
% src.precond.subspace_iter_plain (NO re-orthogonalization inside the loop);
% the single orthonormalization is the pivoted QR inside the capture metric.
% As a stability ablation, the gaussian block additionally runs inverse_reorth
% (src.precond.subspace_iter): the same q exact solves but with orth after
% EVERY solve, in both systems.
%
% Four starting blocks are compared in two matched-width ablation pairs, all
% built from that system's own tentative prolongator Pt (aggregation on the
% sparsified Tsym for the Tsym row, on the sparse A directly for the A row):
%   width m   : gaussian      = randn(n,m)       vs  sketched_tent = Pt*randn(nc,m)
%   width nc  : gaussian_tent = randn(n,nc)      vs  tent          = full(Pt)
% (nc >= m is the raw tentative width, per system.)  This isolates "tentative
% vs Gaussian" at each width.
%
% Two POLYNOMIAL-FILTER families are overlaid on the same panels (only for
% the width-m blocks gaussian / sketched_tent, in BOTH systems), so all
% methods share one x-axis q = number of operator applications:
%   chebyshev : src.precond.chebyshev_apply  -- high-pass T_q on the forward op (q matvecs)
%   power_iz  : src.precond.min_subspace_iter -- (I - Op/lam_max)^q             (q matvecs)
% For the inverse method q is the iteration power (q exact solves); for the
% polynomial filters q is the degree.  This contrasts the fast-but-unstable
% exact inverse against the slower-but-stable polynomial filters.
%
% Capture metric (basis-invariant): directed principal-angle sines from
% span(V_true) into span(Q), via the local subspace_capture_directed.
%
% Downstream PCG (pcg_iters panels): both rows solve the SAME original system
% A x = b; only the preconditioner differs:
%   Tsym row : split two-level operator B = L^-T P L^-1, P built on Ahat = Tsym
%              from the captured basis Q (src.precond.deflation_P_apply)
%   A row    : deflation-only operator B = P, P built directly on A
% The exact-deflation (V_true) iteration count of each row is drawn as a
% dashed floor in its panel.
%
% Outputs (subspace_capture/output_inverse_iter/):
%   results.mat / results.csv        (rows carry a `system` column: Tsym | A)
%   eigspace_err2_inv[_A].pdf           — ||(I-P)Q_true||_2 (log y) vs iteration q
%   angle_capture_fraction_inv[_A].pdf  — fraction of directions with
%                                         sin(theta)<1% vs iteration q
%   kappa_ratio_inv[_A].pdf             — kappa_approx/kappa_exact of the
%                                         two-level deflated system (log y) vs
%                                         iteration q, via the local
%                                         deflated_cond_two_level (kappa_exact =
%                                         deflation with V_true; analytically
%                                         lam_max/lam_cut)
%   pcg_iters_inv[_A].pdf               — downstream PCG iterations to converge
%                                         vs iteration q (see above)
%   aggregate_2x4.png               — all panels, row 1 = Tsym, row 2 = A
%
% Ground truth (smallest k eigenvectors of each operator) is read from the
% caches that run_subspace_capture.m already populated
% (output/cache/eigsTsym_k500.mat, output/cache/eigsA_k500.mat); no
% eigendecomposition is recomputed if those caches are present.
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

% Downstream PCG (split two-level scheme) configuration.  The KL-noise RHS
% and solver tolerances mirror run_krylov_capture.m so the iteration counts
% are directly comparable to the sibling study.
Kmodes    = 50;                  % # KL cosine modes in the solver RHS
sigma     = 1;                   % RHS scale (irrelevant to the Krylov subspace)
bbox      = [-1 1 -1 1];         % unit-sphere x,y extent for the cosine modes
pcg_tol   = 1e-8;                % PCG convergence tolerance
pcg_maxit = 400;                 % PCG iteration cap

P0_kinds = {'gaussian', 'sketched_tent', 'tent', 'gaussian_tent'};

% Methods sharing the common x-axis q (= operator applications):
%   inverse   : plain subspace iteration with Z = Op^{-1}  (q exact solves)
%   chebyshev : Chebyshev high-pass filter on the forward op (q matvecs)
%   power_iz  : damped power filter (I - Op/lam_max)^q       (q matvecs)
% The polynomial filters are overlaid only on the two width-m blocks.
poly_blocks = {'gaussian', 'sketched_tent'};

% Reorth ablation of the exact-inverse method (inverse_reorth =
% src.precond.subspace_iter, orth after every solve), gaussian block only.
reorth_blocks = {'gaussian'};

%% --- Build baseline snapshot A + L ----------------------------------------
fprintf('\n--- Building sphere snapshot (h0=%.4g, %s) ---\n', h0, mesh_method);
[A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
n  = msh.numIN;
Lt = L';
fprintf('A: %d x %d, nnz=%d, sym=%d   nnz(L)=%d\n', ...
        size(A,1), size(A,2), nnz(A), issymmetric(A), nnz(L));

%% --- Ground-truth small eigenpairs of Tsym and A (read from existing caches)
[V_true_T, lam_cut_T, ~] = load_or_compute_eigs_Tsym(cacheDir, A, L, k);
fprintf('V_true_T (smallest %d eigvecs of Tsym): %d x %d  (lam_cut=%.4e)\n', ...
        k, size(V_true_T,1), size(V_true_T,2), lam_cut_T);
[V_true_A, lam_cut_A, ~] = load_or_compute_eigs_A(cacheDir, A, k);
fprintf('V_true_A (smallest %d eigvecs of A):    %d x %d  (lam_cut=%.4e)\n', ...
        k, size(V_true_A,1), size(V_true_A,2), lam_cut_A);

%% --- Operators -------------------------------------------------------------
% One Cholesky factorization of A serves both exact inverses:
%   Tsym^{-1} = L^T A^{-1} L   and   A^{-1}.
dA         = decomposition(A, 'chol');
invApply_T = @(X) Lt * (dA \ (L * X));
invApply_A = @(X) dA \ X;
% Forward operators (for the polynomial-filter methods and deflation builds).
Zfun_Tsym  = @(X) L \ (A * (Lt \ X));
Zfun_A     = @(X) A * X;
% Top of each spectrum (reused from cache) -- needed by the poly filters.
lam_max_T = load_or_compute_lam_max(cacheDir, 'Tsym', Zfun_Tsym, n);
lam_max_A = load_or_compute_lam_max(cacheDir, 'A', A, n);
fprintf('lam_max(Tsym) = %.4e   lam_max(A) = %.4e\n', lam_max_T, lam_max_A);

% KL-noise RHS for the downstream split two-level PCG solve (mirrors
% run_krylov_capture.m:101-107).  (P)CG normalizes the RHS internally, so
% sigma does not affect the iteration count; the seed fixes the realization.
kvec  = generate_kvec(Kmodes);
Phi   = src.forcing.eval_cosine_modes(msh.p, kvec, bbox);   % numIN x Kmodes
rng(seed);
z     = randn(Kmodes, 1);
b_pcg = sigma * sqrt(dt) * (msh.D_II * (Phi * z));           % exact solver rhsI
fprintf('PCG RHS b: Kmodes=%d, seed=%d, ||b||=%.4e\n', Kmodes, seed, norm(b_pcg));

%% --- Tentative prolongators (one per system) --------------------------------
% Tsym must first be materialized and sparsified; A is already sparse.
T_sparse = build_Tsym_sparse(L, Lt, A, drop_rel_tol);
fprintf(['Tsym sparsification: nnz=%d (%.2f%% of n^2), ', ...
         'max|T|=%.3e\n'], ...
        nnz(T_sparse), 100 * nnz(T_sparse) / n^2, ...
        max(abs(nonzeros(T_sparse))));
[Pt_T, nc_T] = build_tent_at_least(T_sparse, theta, maxAggSize, m);
fprintf('Pt_T: nc=%d  (m=%d, maxAggSize=%d)\n', nc_T, m, maxAggSize);
[Pt_A, nc_A] = build_tent_at_least(A, theta, maxAggSize, m);
fprintf('Pt_A: nc=%d  (m=%d, maxAggSize=%d)\n', nc_A, m, maxAggSize);

%% --- The two systems (identical methodology, side-by-side comparison) ------
% makeB wraps the deflation projector P (built on that system's operator with
% tau = its lam_cut) into pcg's preconditioner for the SHARED solve A x = b:
%   Tsym : split two-level  B = L^-T P L^-1
%   A    : deflation-only   B = P
systems = struct( ...
    'name',       {'Tsym',                          'A'}, ...
    'zlabel',     {'Tsym^{-1}',                     'A^{-1}'}, ...
    'Zfun',       {Zfun_Tsym,                       Zfun_A}, ...
    'invApply',   {invApply_T,                      invApply_A}, ...
    'V_true',     {V_true_T,                        V_true_A}, ...
    'lam_cut',    {lam_cut_T,                       lam_cut_A}, ...
    'lam_max',    {lam_max_T,                       lam_max_A}, ...
    'Pt',         {Pt_T,                            Pt_A}, ...
    'nc',         {nc_T,                            nc_A}, ...
    'makeB',      {@(Papply) @(r) Lt \ (Papply(L \ r)), @(Papply) @(r) Papply(r)}, ...
    'tag_suffix', {'',                              '_A'}, ...
    'kappa_exact', {NaN, NaN}, 'pcg_iters_exact', {NaN, NaN});

% Exact floors per system: condition number and downstream PCG iterations of
% the EXACTLY deflated system (W = V_true); every sweep point is reported
% relative to them.
condOpts  = struct('eigs_tol', 1e-8, 'eigs_maxit', 5000);   % per-point (raw W)
exactOpts = struct('eigs_tol', 1e-8, 'eigs_maxit', 5000, 'W_is_orth', true);
for is = 1:numel(systems)
    sys = systems(is);
    exactCond = deflated_cond_two_level(sys.V_true, sys.Zfun, sys.invApply, ...
                                        sys.lam_cut, n, exactOpts);
    if ~exactCond.ok
        error('run_inverse_subspace_iter:kappaExactFailed', ...
              '[%s] kappa_exact computation failed: %s', sys.name, exactCond.err);
    end
    systems(is).kappa_exact = exactCond.kappa;
    systems(is).pcg_iters_exact = two_level_pcg_iters( ...
        sys.V_true, A, sys.Zfun, sys.lam_cut, sys.makeB, ...
        b_pcg, pcg_tol, pcg_maxit);
    fprintf(['[%-4s] kappa_exact = %.6e  (analytic lam_max/lam_cut = %.6e)', ...
             '   pcg_iters_exact = %g\n'], ...
            sys.name, systems(is).kappa_exact, sys.lam_max / sys.lam_cut, ...
            systems(is).pcg_iters_exact);
end

%% --- Pre-build the four P0 blocks per system -------------------------------
% rng(seed) before every build: the width-m 'gaussian' block is bit-identical
% across the two systems; the tent-width blocks differ only through Pt/nc.
P0_cache = struct();
for is = 1:numel(systems)
    sysName = systems(is).name;
    for ik = 1:numel(P0_kinds)
        kind = P0_kinds{ik};
        rng(seed);
        [P0, ncols] = build_P0(kind, systems(is).Pt, systems(is).nc, n, m);
        P0_cache.(sysName).(kind).P  = P0;
        P0_cache.(sysName).(kind).nc = ncols;
        fprintf('P0 [%-4s] %-14s: %d x %d\n', sysName, kind, size(P0,1), ncols);
    end
end

%% --- Sweep ----------------------------------------------------------------
rows = [];
for is = 1:numel(systems)
    sys = systems(is);
    % Operator handles + spectral bounds shared by this system's per-q runs.
    ops = struct('invApply', sys.invApply, 'Zfun', sys.Zfun, ...
                 'lam_cut', sys.lam_cut, 'lam_max', sys.lam_max, 'n', n, ...
                 'kappa_exact', sys.kappa_exact, 'condOpts', condOpts, ...
                 'A', A, 'makeB', sys.makeB, ...
                 'b_pcg', b_pcg, 'pcg_tol', pcg_tol, 'pcg_maxit', pcg_maxit);

    for ik = 1:numel(P0_kinds)
        kind  = P0_kinds{ik};
        P0    = P0_cache.(sys.name).(kind).P;
        ncols = P0_cache.(sys.name).(kind).nc;

        % Inverse iteration for every block; reorth ablation on the gaussian
        % block; polynomial filters only on the width-m blocks.
        all_methods = {'inverse', 'inverse_reorth', 'chebyshev', 'power_iz'};
        is_poly     = any(strcmp(kind, poly_blocks));
        methods     = all_methods([true, any(strcmp(kind, reorth_blocks)), ...
                                   is_poly, is_poly]);

        for im = 1:numel(methods)
            method = methods{im};
            for iq = 1:numel(iters)
                q    = iters(iq);
                info = run_one_method(method, ops, P0, q, sys.V_true);
                info.system   = sys.name;
                info.P0_kind  = kind;
                info.P0_ncols = ncols;
                info.method   = method;
                info.iter     = q;
                rows = [rows; info];                                  %#ok<AGROW>
                fprintf(['  [%-4s] %-14s %-10s q=%2d : err_2=%.3e', ...
                         '  angle_capture=%.3f  kappa_ratio=%.3e', ...
                         '  pcg_iters=%g\n'], ...
                        sys.name, kind, method, q, info.eigspace_err_2, ...
                        info.angle_capture_frac_1pct, info.kappa_ratio, ...
                        info.pcg_iters);
            end
        end
    end
end

%% --- Save -----------------------------------------------------------------
meta = struct('n', n, 'k', k, 'm', m, 'iters', iters, ...
              'contrast', contrast, 'h0', h0, 't_snap', t_snap, ...
              'systems', {{systems.name}}, ...
              'nc_T', nc_T, 'nc_A', nc_A, ...
              'lam_cut_T', lam_cut_T, 'lam_max_T', lam_max_T, ...
              'lam_cut_A', lam_cut_A, 'lam_max_A', lam_max_A, ...
              'kappa_exact_T', systems(1).kappa_exact, ...
              'kappa_exact_A', systems(2).kappa_exact, ...
              'pcg_iters_exact_T', systems(1).pcg_iters_exact, ...
              'pcg_iters_exact_A', systems(2).pcg_iters_exact, ...
              'Kmodes', Kmodes, 'seed', seed, ...
              'pcg_tol', pcg_tol, 'pcg_maxit', pcg_maxit);
save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');
write_results_csv(fullfile(outDir, 'results.csv'), rows);
fprintf('\nresults.mat and results.csv written to:\n  %s\n', outDir);

%% --- Render the plots -------------------------------------------------------
fprintf('\n--- Rendering plots ---\n');
sysPlot = struct('name', {systems.name}, 'zlabel', {systems.zlabel}, ...
                 'pcg_iters_exact', {systems.pcg_iters_exact}, ...
                 'tag_suffix', {systems.tag_suffix});
make_inverse_plots(rows, outDir, sysPlot);

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
    t0   = tic;
    % Name-value options, NOT the legacy opts struct: eigs silently ignores
    % struct fields with these capitalized names.
    [Vraw, Dmat] = eigs(Tinv, size(A, 1), k + 1, 'smallestabs', ...
                        'Tolerance', 1e-10, 'MaxIterations', 5000, ...
                        'IsFunctionSymmetric', true);
    [D, idx]     = sort(real(diag(Dmat)), 'ascend');
    V            = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_true    = V(:, 1:k);
    lam_cut   = D(k + 1);
    lam_first = D(1);
end

function [V_true, lam_cut, lam_first] = load_or_compute_eigs_A(cacheDir, A, k)
%LOAD_OR_COMPUTE_EIGS_A  Smallest k+1 eigenpairs of the original matrix A.
%   Same cache convention as run_subspace_capture.m (eigsA_k%d.mat).  Returns
%   first k vecs, the cutoff D(k+1), and the smallest eigenvalue D(1).
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
    t0 = tic;
    % Name-value options, NOT the legacy opts struct: eigs silently ignores
    % struct fields with these capitalized names.
    [Vraw, Dmat] = eigs(A, k + 1, 'smallestabs', ...
                        'Tolerance', 1e-10, 'MaxIterations', 5000);
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

function [P0, ncols] = build_P0(kind, Pt, nc, n, m)
%BUILD_P0  Construct one of the four starting blocks from a system's tentative.
    switch kind
        case 'gaussian'                 % width m
            P0    = randn(n, m);
            ncols = m;
        case 'sketched_tent'            % width m
            G     = randn(nc, m);
            P0    = Pt * G;
            ncols = m;
        case 'tent'                     % width nc (raw tentative)
            P0    = full(Pt);
            ncols = nc;
        case 'gaussian_tent'            % width nc (Gaussian matched to raw tentative)
            P0    = randn(n, nc);
            ncols = nc;
        otherwise
            error('build_P0: unknown kind %s', kind);
    end
end

function info = run_one_method(method, ops, P0, q, V_true)
%RUN_ONE_METHOD  Apply one method at "work level" q, then measure capture.
%   ops carries one system's operators (Zfun forward, invApply exact inverse)
%   and spectral bounds; the same code path serves Tsym and A.
%   q counts operator applications for all methods:
%     inverse        : q plain applications of Z = Op^{-1} (no reorth).
%     inverse_reorth : q applications of Z = Op^{-1} with orth after EVERY
%                      solve (src.precond.subspace_iter); output orthonormal.
%     chebyshev      : degree-q Chebyshev high-pass filter on the forward op.
%     power_iz       : degree-q damped power filter (I - Op/lam_max)^q (final
%                      orth done inside min_subspace_iter).
%   inverse/chebyshev pass the RAW block: the metric's pivoted QR is the
%   single orthonormalization.  At q=0 all methods reduce to orth-of-P0 -- a
%   built-in consistency check.
    info = new_capture_info();
    t0   = tic;
    try
        switch method
            case 'inverse'
                Q = src.precond.subspace_iter_plain(ops.invApply, P0, q);
                Q_is_orth = false;
            case 'inverse_reorth'
                % subspace_iter orths the start block and after every apply.
                Q = src.precond.subspace_iter(ops.invApply, P0, q);
                Q_is_orth = true;
            case 'chebyshev'
                Q = src.precond.chebyshev_apply(ops.Zfun, P0, q, ...
                                                ops.lam_cut, ops.lam_max);
                Q_is_orth = false;
            case 'power_iz'
                Dinv = (1 / ops.lam_max) * ones(ops.n, 1);
                % min_subspace_iter ends with orth(Y) => output orthonormal.
                Q = src.precond.min_subspace_iter(ops.Zfun, P0, q, ...
                                                  Dinv, 1.0, false);
                Q_is_orth = true;
            otherwise
                error('run_one_method: unknown method %s', method);
        end
        info = fill_capture_info(info, V_true, Q, Q_is_orth);
        % Two-level deflated condition number with W = orth(range(Q)); a
        % failed estimate leaves NaN without invalidating the capture row.
        c = deflated_cond_two_level(Q, ops.Zfun, ops.invApply, ...
                                    ops.lam_cut, ops.n, ops.condOpts);
        info.kappa_approx = c.kappa;
        info.kappa_ratio  = c.kappa / ops.kappa_exact;
        info.r_defl       = c.r;

        % Downstream cost: iterations of the deflated PCG on A x = b using the
        % captured basis Q as the deflation set -- the empirical companion of
        % kappa_ratio.  P is built on this system's operator; ops.makeB wraps
        % it into the preconditioner (split L^-T P L^-1 for Tsym, plain P for
        % A).  A poorly captured Q (non-SPD coarse matrix) throws inside
        % deflation_P_apply and is caught below, leaving pcg_iters = NaN
        % without dropping the row.
        info.pcg_iters = two_level_pcg_iters( ...
            Q, ops.A, ops.Zfun, ops.lam_cut, ops.makeB, ...
            ops.b_pcg, ops.pcg_tol, ops.pcg_maxit);
    catch ME
        info.err = regexprep(ME.message, '\n.*', '');
        warning('run_inverse_subspace_iter:run_one_method_failed', ...
                'method=%s q=%d failed: %s', method, q, info.err);
    end
    info.time_seconds = toc(t0);
end

function it = two_level_pcg_iters(W, A, Zfun, tau, makeB, b, tol, maxit)
%TWO_LEVEL_PCG_ITERS  PCG iterations for A x = b with a deflation preconditioner.
%   Runs pcg on the ORIGINAL system A x = b (pcg's matrix arg is @(X) A*X, RHS
%   is the raw b).  The deflation operator P is BUILT on the system operator
%   Zfun from the deflation basis span(W) (src.precond.deflation_P_apply,
%   tau = that system's lam_cut); makeB(Papply) wraps it into pcg's
%   preconditioner (5th arg):
%     Tsym row : makeB = @(P) @(r) Lt \ (P(L \ r))  -- split B = L^-T P L^-1,
%                the exact scheme used by solve_two_level in the repo's
%                benchmarks (+src/+solver/solve_deflate_M_P.m)
%     A row    : makeB = @(P) @(r) P(r)             -- deflation-only B = P
%
%   Returns the pcg iteration count (4th output of pcg), or NaN if the coarse
%   matrix W'*Op*W is not numerically SPD (deflation_P_apply throws on the
%   chol) or pcg errors -- a failure never invalidates the capture row.
    try
        Wd     = orth(full(W));                     % orthonormal deflation basis
        Papply = src.precond.deflation_P_apply(Wd, Zfun, tau);
        Bapply = makeB(Papply);
        [~, ~, ~, it] = pcg(@(X) A * X, b, tol, maxit, @(r) Bapply(r));
    catch
        it = NaN;
    end
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
    % Name-value options, NOT the legacy opts struct: eigs silently ignores
    % struct fields with these capitalized names.
    if isnumeric(op)
        d = eigs(op, 1, 'largestabs', ...
                 'Tolerance', 1e-8, 'MaxIterations', 5000);
    else
        d = eigs(op, n, 1, 'largestabs', ...
                 'Tolerance', 1e-8, 'MaxIterations', 5000, ...
                 'IsFunctionSymmetric', true);
    end
    lam_max = real(d);
    save(cachePath, 'lam_max', '-v7');
end

function info = new_capture_info()
%NEW_CAPTURE_INFO  Empty result struct.
%   All metrics are the basis-invariant directed principal-angle ones.
    info = struct( ...
        'eigspace_err_2', NaN, 'eigspace_err_fro', NaN, ...
        'angle_capture_frac_1pct', NaN, ...
        'n_angle_below_1pct', NaN, 'n_angle_below_0p1pct', NaN, ...
        'r_true', NaN, 'r_comp', NaN, ...
        'kappa_approx', NaN, 'kappa_ratio', NaN, 'r_defl', NaN, ...
        'pcg_iters', NaN, ...
        'time_seconds', NaN, 'ok', false, 'err', '');
end

function info = fill_capture_info(info, V_true, Q, Q_is_orth)
%FILL_CAPTURE_INFO  Populate capture metrics from a computed basis Q.
%   V_true comes from eigs (symmetric problem) => orthonormal columns;
%   Q_is_orth says whether Q is already orthonormal too.
    capOpts = struct('true_is_orth', true, 'comp_is_orth', Q_is_orth);
    cap = subspace_capture_directed(V_true, Q, [], capOpts);
    info.eigspace_err_2          = cap.eigspace_err_2;
    info.eigspace_err_fro        = cap.eigspace_err_fro;
    info.n_angle_below_1pct      = cap.n_angle_below_1pct;
    info.n_angle_below_0p1pct    = cap.n_angle_below_0p1pct;
    info.angle_capture_frac_1pct = cap.n_angle_below_1pct / max(cap.r_true, 1);
    info.r_true                  = cap.r_true;
    info.r_comp                  = cap.r_comp;
    info.ok = true;
end

function write_results_csv(csvPath, rows)
%WRITE_RESULTS_CSV  Flat CSV of the sweep results (directed-angle metrics).
    fid = fopen(csvPath, 'w');
    fprintf(fid, ['system,P0_kind,method,P0_ncols,iter,', ...
                  'eigspace_err_2,eigspace_err_fro,angle_capture_frac_1pct,', ...
                  'n_angle_below_1pct,n_angle_below_0p1pct,r_true,r_comp,', ...
                  'kappa_approx,kappa_ratio,r_defl,pcg_iters,time_seconds\n']);
    for i = 1:numel(rows)
        r = rows(i);
        fprintf(fid, '%s,%s,%s,%d,%d,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n', ...
                r.system, r.P0_kind, r.method, r.P0_ncols, r.iter, ...
                r.eigspace_err_2, r.eigspace_err_fro, ...
                r.angle_capture_frac_1pct, ...
                r.n_angle_below_1pct, r.n_angle_below_0p1pct, ...
                r.r_true, r.r_comp, ...
                r.kappa_approx, r.kappa_ratio, r.r_defl, ...
                r.pcg_iters, r.time_seconds);
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

function kvec = generate_kvec(K)
%GENERATE_KVEC  K wavenumber pairs (kx,ky) sorted by ascending frequency.
%   Copied verbatim from run_krylov_capture.m / solve_deflate_M_P.m so the RHS
%   matches the solver.
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
%% Rendering: per-system panels (+ 2x4 aggregate PNG)
%% =========================================================================
function make_inverse_plots(rows, out_dir, sysPlot)
%MAKE_INVERSE_PLOTS  Four capture/cost panels per system vs iteration count.
%   sysPlot is a struct array (name, zlabel, pcg_iters_exact, tag_suffix), one
%   entry per system.  Each system gets its own four PDFs (the Tsym suffix is
%   '' so the historical filenames are unchanged; the A row appends '_A'), and
%   the aggregate PNG is a 2x4 grid: row 1 = Tsym, row 2 = A.
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    specs = struct( ...
        'metric', {'eigspace_err_2',        'angle_capture_frac_1pct',              'kappa_ratio',                      'pcg_iters'}, ...
        'yscale', {'log',                   'linear',                               'log',                              'linear'}, ...
        'ylabel', {'||(I-P)Q_{true}||_2',   'captured directions (sin\theta < 1%)', '\kappa_{approx} / \kappa_{exact}', 'PCG iterations to converge'}, ...
        'title',  {'eigenspace error',      'angle capture fraction',               'deflated condition-number ratio',  'downstream PCG cost'}, ...
        'legloc', {'southwest',             'northeast',                            'northeast',                        'northeast'}, ...
        'tag',    {'eigspace_err2_inv',     'angle_capture_fraction_inv',           'kappa_ratio_inv',                  'pcg_iters_inv'});

    for is = 1:numel(sysPlot)
        sp     = sysPlot(is);
        rows_s = rows(strcmp({rows.system}, sp.name));
        for ip = 1:numel(specs)
            fig = figure('Visible', 'off', 'Units', 'inches', ...
                         'Position', [0 0 5.4 3.4], 'Color', 'w');
            draw_inverse_panel(axes(fig), rows_s, specs(ip), ...
                               sp.pcg_iters_exact, sp.zlabel);
            outfile = fullfile(out_dir, [specs(ip).tag sp.tag_suffix '.pdf']);
            exportgraphics(fig, outfile, 'ContentType', 'vector');
            close(fig);
            fprintf('Wrote %s\n', outfile);
        end
    end

    % Convenience 2x4 PNG: one row per system, metrics aligned column-wise.
    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 21 7.2], 'Color', 'w');
    tl = tiledlayout(fig, numel(sysPlot), numel(specs), ...
                     'Padding', 'compact', 'TileSpacing', 'compact');
    for is = 1:numel(sysPlot)
        sp     = sysPlot(is);
        rows_s = rows(strcmp({rows.system}, sp.name));
        for ip = 1:numel(specs)
            draw_inverse_panel(nexttile(tl), rows_s, specs(ip), ...
                               sp.pcg_iters_exact, sp.zlabel);
        end
    end
    title(tl, ['Capture vs operator applications q: exact-inverse iteration ', ...
               'vs polynomial filters (row 1: Tsym, row 2: original A)'], ...
          'FontWeight', 'bold', 'FontSize', 12);
    outfile = fullfile(out_dir, 'aggregate_2x4.png');
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
    fprintf('Wrote %s\n', outfile);
end

function draw_inverse_panel(ax, rows, spec, pcg_iters_exact, zlabel_str)
%DRAW_INVERSE_PANEL  One (metric vs iteration) panel overlaying the series.
%   rows must already be filtered to a single system.  pcg_iters_exact draws
%   that system's exact-deflation floor in the pcg_iters panel; zlabel_str
%   names the system's inverse operator in the title.
    if nargin < 4, pcg_iters_exact = NaN; end
    if nargin < 5, zlabel_str = 'Tsym^{-1}'; end
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

    if strcmp(spec.metric, 'kappa_ratio')
        yline(ax, 1, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    elseif strcmp(spec.metric, 'pcg_iters') && isfinite(pcg_iters_exact)
        yline(ax, pcg_iters_exact, '--', ...
              sprintf('exact-deflation floor (%g)', pcg_iters_exact), ...
              'Color', [0.4 0.4 0.4], 'LineWidth', 1.0, 'FontSize', 7, ...
              'LabelHorizontalAlignment', 'left', ...
              'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    end

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
    elseif strcmp(spec.metric, 'pcg_iters')
        % Pad the y-range so the exact-deflation floor line stays on-screen.
        yv = [yall(isfinite(yall)), pcg_iters_exact];
        yv = yv(isfinite(yv));
        if ~isempty(yv)
            lo = min(yv);  hi = max(yv);  pad = 0.05 * max(hi - lo, 1);
            ylim(ax, [max(0, lo - pad), hi + pad]);
        end
    elseif strcmp(spec.yscale, 'log') && ~isempty(yall)
        yp = yall(yall > 0);
        if ~isempty(yp)
            ylim(ax, [min(yp) * 0.5, max(yp) * 1.5]);
        end
    end

    xlabel(ax, 'number of subspace iterations q', ...
           'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, spec.ylabel, 'FontSize', 10, 'FontWeight', 'bold');
    title(ax, sprintf('%s  (Z = %s)', spec.title, zlabel_str), ...
          'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');

    lgd = legend(ax, handles(keep), legs(keep), ...
                 'Location', spec.legloc, 'Box', 'on', ...
                 'EdgeColor', [0.65 0.65 0.65], 'Color', 'white', ...
                 'FontSize', 6.5, 'NumColumns', 2, 'Interpreter', 'none');
    lgd.ItemTokenSize = [14, 6];
end

function series = series_spec()
%SERIES_SPEC  The 9 (block, method) series with consistent visuals.
%   Shared by both system rows (kind names are system-agnostic; the panel
%   title carries the system).  Color encodes the starting block; line-style +
%   marker encode the method.  Polynomial filters (chebyshev/power_iz) only
%   exist for the two width-m blocks.
    block_color = struct( ...
        'gaussian',      [0.00 0.00 0.00], ...
        'sketched_tent', [0.85 0.33 0.10], ...
        'tent',          [0.20 0.70 0.30], ...
        'gaussian_tent', [0.35 0.35 0.75]);
    block_short = struct( ...
        'gaussian',      'gaussian(m)', ...
        'sketched_tent', 'sk_tent(m)', ...
        'tent',          'tent(nc)', ...
        'gaussian_tent', 'gauss_tent(nc)');
    method_style = struct('inverse', '-',  'inverse_reorth', '-.', ...
                          'chebyshev', '--', 'power_iz', ':');
    method_mark  = struct('inverse', 'o',  'inverse_reorth', 'd', ...
                          'chebyshev', 's',  'power_iz', '^');

    % (block, method) pairs in legend order.
    combos = { ...
        'gaussian',      'inverse'; ...
        'gaussian',      'inverse_reorth'; ...
        'gaussian',      'chebyshev'; ...
        'gaussian',      'power_iz'; ...
        'sketched_tent', 'inverse'; ...
        'sketched_tent', 'chebyshev'; ...
        'sketched_tent', 'power_iz'; ...
        'tent',          'inverse'; ...
        'gaussian_tent', 'inverse'};

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
