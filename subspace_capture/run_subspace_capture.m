% RUN_SUBSPACE_CAPTURE  One-shot ablation subspace-capture study on a single
% ball_surface / latitude_banding snapshot.
%
% Combines what used to be three separate scripts into one program over a
% single snapshot A:
%   1. the polynomial-filter capture sweep  (-> results.mat / results.csv),
%   2. four paper capture PDFs + one 2x2 aggregate PNG (-> ablation_paper/),
%   3. the five spectrum / spy diagnostics   (-> spectrum_spy_paper/*.pdf).
%
% The snapshot A is built from the unit-sphere surface mesh with a
% latitude-banding kappa at contrast C = 60 and t = 0.
%
% Three polynomial families are compared for capturing the smallest k
% eigenvectors of an operator Z:
%   1) Plain damped power     : (I - Z/lam_max)^i P0   (src.precond.min_subspace_iter)
%   2) Chebyshev high-pass     : T_i(Zbar) P0          (src.precond.chebyshev_apply)
%   3) Balanced Chebyshev      : rho_d*(Zbar) P0, Lanczos sigma-damped, left-end
%      centered, with the degree d* chosen AUTOMATICALLY (paper-optimal: the
%      smallest degree whose filter value at the band edge falls below phi=0.3).
%      Saad, arXiv:1512.08135; local make_balanced_cheb_filter + apply_cheb_filter.
%      Reported as a single optimal operating point per (Z, P0_kind), not swept.
%
% Sweep:
%   Z       : { A,  T_sym = L^{-1} A L^{-T} }   (L = ichol(A,'nofill'))
%   P0_kind : { gaussian, sketched_tent_A,  (+ sketched_tent_T for Z=Tsym) }
%   poly    : { power_iz, chebyshev } swept over degrees below;
%             chebyshev_balanced = one auto-degree optimal point per series.
%   degree  : [0 1 2 3 5 8 12 16 20 30 50]   (power_iz / chebyshev only)
%
% Spectral bounds are exact: lam_first = D(1) (left end), lam_cut = D(k+1)
% (stop-band edge) from the cached eigendecomposition, lam_max from
% eigs(..., 'largestabs').  Caches go in output/cache/.
%
% Usage:
%   cd subspace_capture
%   run_subspace_capture

thisFileDir = fileparts(mfilename('fullpath'));
repoRoot    = fileparts(thisFileDir);
addpath(repoRoot);     % so `src.*` packages resolve
addpath(thisFileDir);  % so the local balanced-Cheb filter files resolve

% All generated deliverables live under one output/ subfolder, so the script
% directory holds nothing but this .m file.
outDir       = fullfile(thisFileDir, 'output');
cacheDir     = fullfile(outDir, 'cache');
out_capture  = fullfile(outDir, 'ablation_paper');
out_spectrum = fullfile(outDir, 'spectrum_spy_paper');
if ~isfolder(outDir),       mkdir(outDir);       end
if ~isfolder(cacheDir),     mkdir(cacheDir);     end
if ~isfolder(out_capture),  mkdir(out_capture);  end
if ~isfolder(out_spectrum), mkdir(out_spectrum); end

%% --- Snapshot configuration (was build_ablation_params + build_ablation_cfg) ---
h0          = 0.05;     % baseline mesh edge length
contrast    = 60;       % kappa_max / kappa_min
t_snap      = 0;        % snapshot time level
dt          = 1;        % was params.dt
Tstep       = 100;      % was params.Tstep
Tmax        = Tstep * dt;     % = 100, band-drift phase scale for kappaFun
mesh_method = 'pdetoolbox';  % 'pdetoolbox' (default, fast) or 'distmesh'
kappa_name  = sprintf('latitude_banding_C%g', contrast);
h0_refine   = [0.10, 0.07, 0.05, 0.035];   % mesh-refinement sweep

%% --- Sweep configuration --------------------------------------------------
k            = 500;
m            = 2 * k;
degrees      = [0 1 2 3 5 8 12 16 20 30 50];
theta        = 0.05;
maxAggSize   = 16;
seed         = 1;
drop_rel_tol = 1e-8;

P0_kinds_A    = {'gaussian', 'sketched_tent_A'};
P0_kinds_Tsym = {'gaussian', 'sketched_tent_A', 'sketched_tent_T'};
poly_kinds    = {'power_iz', 'chebyshev', 'chebyshev_balanced'};
Z_kinds       = {'A', 'Tsym'};

% Balanced Chebyshev filter (Saad, arXiv:1512.08135). The paper uses Lanczos
% sigma-damping for all polynomial filters and an end-interval boundary
% threshold phi = 0.3, with the degree chosen AUTOMATICALLY (smallest degree
% whose filter value at the band edge drops below phi). Both exposed for A/B
% testing.
cheb_bal_damping = 'sigma';
cheb_bal_phi     = 0.3;

%% --- Build baseline snapshot A + L ----------------------------------------
fprintf('\n--- Building sphere snapshot (h0=%.4g, %s) ---\n', h0, mesh_method);
[A, L, msh] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
n  = msh.numIN;
Lt = L';
fprintf('A: %d x %d, nnz=%d, sym=%d   nnz(L)=%d\n', ...
        size(A,1), size(A,2), nnz(A), issymmetric(A), nnz(L));

%% --- Operator handles -----------------------------------------------------
Zfun_A    = @(X) A * X;
Zfun_Tsym = @(X) L \ (A * (Lt \ X));

%% --- Cached lam_max for A and T_sym ---------------------------------------
lam_max_A = load_or_compute_lam_max(cacheDir, 'A',    A,         n);
lam_max_T = load_or_compute_lam_max(cacheDir, 'Tsym', Zfun_Tsym, n);

%% --- Cached ground-truth small eigenpairs of A and T_sym ------------------
[V_true_A, lam_cut_A, lam_first_A] = load_or_compute_eigs_A(cacheDir, A, k);
[V_true_T, lam_cut_T, lam_first_T] = load_or_compute_eigs_Tsym(cacheDir, A, L, k);

fprintf('Z=A    : lam_1 = %.4e, lam_cut(k+1) = %.4e, lam_max = %.4e\n', ...
        lam_first_A, lam_cut_A, lam_max_A);
fprintf('Z=Tsym : lam_1 = %.4e, lam_cut(k+1) = %.4e, lam_max = %.4e\n', ...
        lam_first_T, lam_cut_T, lam_max_T);

%% --- Sparsified T_sym for sketched_tent_T --------------------------------
T_sparse = build_Tsym_sparse(L, Lt, A, drop_rel_tol);
fprintf(['Tsym sparsification: nnz=%d (%.2f%% of n^2), ', ...
         'max|T|=%.3e, cutoff=%.3e\n'], ...
        nnz(T_sparse), 100 * nnz(T_sparse) / n^2, ...
        max(abs(nonzeros(T_sparse))), ...
        drop_rel_tol * max(abs(nonzeros(T_sparse))));

%% --- Tentative prolongators (nc >= m) -------------------------------------
[Pt_A, nc_A] = build_tent_at_least(A,        theta, maxAggSize, m);
[Pt_T, nc_T] = build_tent_at_least(T_sparse, theta, maxAggSize, m);
fprintf('Pt_A: nc=%d  (m=%d, maxAggSize=%d)\n', nc_A, m, maxAggSize);
fprintf('Pt_T: nc=%d  (m=%d, maxAggSize=%d)\n', nc_T, m, maxAggSize);

%% --- Pre-build all P0s ----------------------------------------------------
all_kinds = union(P0_kinds_A, P0_kinds_Tsym, 'stable');
P0_cache  = struct();
for ik = 1:numel(all_kinds)
    kind = all_kinds{ik};
    rng(seed);
    [P0, ncols] = build_P0(kind, Pt_A, nc_A, Pt_T, nc_T, n, m);
    P0_cache.(kind).P  = P0;
    P0_cache.(kind).nc = ncols;
    fprintf('P0 %-17s: %d cols\n', kind, ncols);
end

%% --- Sweep ---------------------------------------------------------------
rows = [];
for iz = 1:numel(Z_kinds)
    Zk = Z_kinds{iz};
    if strcmp(Zk, 'A')
        Zfun    = Zfun_A;
        bnds    = struct('lam_first', lam_first_A, 'lam_cut', lam_cut_A, ...
                         'lam_max', lam_max_A, 'damping', cheb_bal_damping, ...
                         'phi', cheb_bal_phi);
        V_true  = V_true_A;
        P0_kinds_for_Z = P0_kinds_A;
    else
        Zfun    = Zfun_Tsym;
        bnds    = struct('lam_first', lam_first_T, 'lam_cut', lam_cut_T, ...
                         'lam_max', lam_max_T, 'damping', cheb_bal_damping, ...
                         'phi', cheb_bal_phi);
        V_true  = V_true_T;
        P0_kinds_for_Z = P0_kinds_Tsym;
    end

    for ik = 1:numel(P0_kinds_for_Z)
        kind  = P0_kinds_for_Z{ik};
        P0    = P0_cache.(kind).P;
        ncols = P0_cache.(kind).nc;

        for ipol = 1:numel(poly_kinds)
            poly = poly_kinds{ipol};

            if strcmp(poly, 'chebyshev_balanced')
                % Paper-optimal: one auto-degree operating point (not swept).
                [info, dstar] = run_balanced_opt(Zfun, P0, bnds, V_true);
                info.Z        = Zk;
                info.P0_kind  = kind;
                info.P0_ncols = ncols;
                info.poly     = poly;
                info.degree   = dstar;
                rows = [rows; info];                                  %#ok<AGROW>
            else
                for id = 1:numel(degrees)
                    i_deg = degrees(id);
                    info  = run_one(Zfun, P0, poly, i_deg, bnds, V_true);
                    info.Z        = Zk;
                    info.P0_kind  = kind;
                    info.P0_ncols = ncols;
                    info.poly     = poly;
                    info.degree   = i_deg;
                    rows = [rows; info];                              %#ok<AGROW>
                end
            end
        end
    end
end

%% --- Save -----------------------------------------------------------------
meta = struct('n', n, 'k', k, 'm', m, 'degrees', degrees, ...
              'contrast', contrast, 'h0', h0, 't_snap', t_snap, ...
              'kappa_name', kappa_name, ...
              'cheb_bal_damping', cheb_bal_damping, 'cheb_bal_phi', cheb_bal_phi, ...
              'lam_first_A', lam_first_A, 'lam_cut_A', lam_cut_A, 'lam_max_A', lam_max_A, ...
              'lam_first_T', lam_first_T, 'lam_cut_T', lam_cut_T, 'lam_max_T', lam_max_T, ...
              'nc_A', nc_A, 'nc_T', nc_T);
save(fullfile(outDir, 'results.mat'), 'rows', 'meta', '-v7');

write_results_csv(fullfile(outDir, 'results.csv'), rows);
fprintf('\nresults.mat and results.csv written to:\n  %s\n', outDir);

%% --- Render the four capture plots + a 2x2 aggregate PNG ------------------
fprintf('\n--- Rendering capture plots ---\n');
make_ablation_paper(rows, out_capture);
make_ablation_aggregate_png(rows, out_capture);

%% --- Spectrum / spy diagnostics -------------------------------------------
% Baseline small-k spectra: read straight from the caches the sweep populated.
D_A_base = load_D_from_cache(fullfile(cacheDir, sprintf('eigsA_k%d.mat', k)), k);
D_M_base = load_D_from_cache(fullfile(cacheDir, sprintf('eigsTsym_k%d.mat', k)), k);

% Baseline large-k spectra (one-time compute + cache).
[~, ~] = load_or_compute_eigs_A_large(cacheDir, A, k);
[~, ~] = load_or_compute_eigs_Tsym_large(cacheDir, A, L, k);
D_A_lg = load_D_from_cache_desc(fullfile(cacheDir, sprintf('eigsA_large_k%d.mat', k)), k);
D_M_lg = load_D_from_cache_desc(fullfile(cacheDir, sprintf('eigsTsym_large_k%d.mat', k)), k);

% Mesh-refinement sweeps (small + large).
nH      = numel(h0_refine);
D_A_all = cell(nH, 1);  D_M_all = cell(nH, 1);  n_all = zeros(nH, 1);
D_A_lg_all = cell(nH, 1);  D_M_lg_all = cell(nH, 1);
for ih = 1:nH
    h0r = h0_refine(ih);
    [D_A_all{ih}, D_M_all{ih}, n_all(ih)] = ...
        load_or_compute_eigs_refine(cacheDir, h0r, contrast, t_snap, dt, Tmax, mesh_method, k);
    [D_A_lg_all{ih}, D_M_lg_all{ih}, ~] = ...
        load_or_compute_eigs_refine_large(cacheDir, h0r, contrast, t_snap, dt, Tmax, mesh_method, k);
end

sd = struct('h0_base', h0, 'n_base', n, 'k', k, 'A', A, 'L', L, ...
            'D_A_base', D_A_base, 'D_M_base', D_M_base, ...
            'D_A_lg', D_A_lg, 'D_M_lg', D_M_lg, ...
            'h0_refine', h0_refine, 'n_all', n_all, ...
            'D_A_all', {D_A_all}, 'D_M_all', {D_M_all}, ...
            'D_A_lg_all', {D_A_lg_all}, 'D_M_lg_all', {D_M_lg_all});

fprintf('\n--- Rendering spectrum / spy plots ---\n');
plot_spectrum_and_spy(sd, out_spectrum);

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
%ASSEMBLE_SNAPSHOT_A  Build A = D_II + dt*K_II at one time level.
%   Closed-sphere case (numBdry = 0, so K_IB is unused).
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

function lam_max = load_or_compute_lam_max(cacheDir, kind, op, n)
%LOAD_OR_COMPUTE_LAM_MAX  Return cached lam_max(Z); compute and cache if missing.
    cachePath = fullfile(cacheDir, sprintf('lam_max_%s.mat', kind));
    if isfile(cachePath)
        S = load(cachePath, 'lam_max');
        lam_max = S.lam_max;
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
        return;
    end
    fprintf('Computing smallest %d eigenpairs of A (one-time)...\n', k + 1);
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    t0   = tic;
    [Vraw, Dmat] = eigs(A, k + 1, 'smallestabs', opts);
    [D, idx]     = sort(real(diag(Dmat)), 'ascend');
    V            = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_true    = V(:, 1:k);
    lam_cut   = D(k + 1);
    lam_first = D(1);
end

function [V_true, lam_cut, lam_first] = load_or_compute_eigs_Tsym(cacheDir, A, L, k)
%LOAD_OR_COMPUTE_EIGS_TSYM  Smallest k+1 eigenpairs of T_sym = L^{-1}AL^{-T}.
%   Uses inverse handle Tinv = Lt*(A\(L*x)) so eigs('smallestabs') becomes
%   a largest-eigenvalues problem on the inverse.  Returns first k vecs, the
%   cutoff D(k+1), and the smallest eigenvalue D(1).
    cachePath = fullfile(cacheDir, sprintf('eigsTsym_k%d.mat', k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_true    = S.V(:, 1:k);
        lam_cut   = S.D(k + 1);
        lam_first = S.D(1);
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

function [V_lg, lam_max] = load_or_compute_eigs_A_large(cacheDir, A, k)
%LOAD_OR_COMPUTE_EIGS_A_LARGE  Largest k eigenpairs of A, sorted descending.
    cachePath = fullfile(cacheDir, sprintf('eigsA_large_k%d.mat', k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_lg    = S.V(:, 1:k);
        lam_max = S.D(1);
        return;
    end
    fprintf('Computing largest %d eigenpairs of A...\n', k);
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    t0   = tic;
    [Vraw, Dmat] = eigs(A, k, 'largestabs', opts);
    [D, idx]     = sort(real(diag(Dmat)), 'descend');
    V            = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_lg    = V(:, 1:k);
    lam_max = D(1);
end

function [V_lg, lam_max] = load_or_compute_eigs_Tsym_large(cacheDir, A, L, k)
%LOAD_OR_COMPUTE_EIGS_TSYM_LARGE  Largest k eigenpairs of M = L^{-1} A L^{-T}.
%   Uses the FORWARD handle Tfun = L\(A*(Lt\x)) -- 'largestabs' on a function
%   handle is well-posed (unlike 'smallestabs' which needs the inverse).
    cachePath = fullfile(cacheDir, sprintf('eigsTsym_large_k%d.mat', k));
    if isfile(cachePath)
        S = load(cachePath, 'V', 'D');
        V_lg    = S.V(:, 1:k);
        lam_max = S.D(1);
        return;
    end
    fprintf('Computing largest %d eigenpairs of L^{-1} A L^{-T}...\n', k);
    Lt   = L';
    Tfun = @(x) L \ (A * (Lt \ x));
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000, ...
                  'IsFunctionSymmetric', true);
    t0   = tic;
    [Vraw, Dmat] = eigs(Tfun, size(A, 1), k, 'largestabs', opts);
    [D, idx]     = sort(real(diag(Dmat)), 'descend');
    V            = real(Vraw(:, idx));
    fprintf('  done in %.1f s\n', toc(t0));
    save(cachePath, 'V', 'D', 'k', '-v7');
    V_lg    = V(:, 1:k);
    lam_max = D(1);
end

function [D_A, D_M, n] = load_or_compute_eigs_refine(cacheDir, h0, contrast, t_snap, dt, Tmax, mesh_method, k)
%LOAD_OR_COMPUTE_EIGS_REFINE  Smallest k eigs of A and Tsym at given h0.
    cachePath = fullfile(cacheDir, ...
        sprintf('eigs_refine_h0_%s_k%d.mat', tag(h0), k));
    if isfile(cachePath)
        S = load(cachePath, 'D_A', 'D_M', 'n');
        D_A = S.D_A; D_M = S.D_M; n = S.n;
        fprintf('h0=%.4g  cached %s (n=%d)\n', h0, cachePath, n);
        return;
    end
    fprintf('\n--- h0=%.4g : building snapshot + eigs ---\n', h0);
    [A, L] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
    n = size(A, 1);

    fprintf('  smallest %d eigs of A...\n', k);
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    t0   = tic;
    [~, Dmat] = eigs(A, k, 'smallestabs', opts);
    D_A = sort(real(diag(Dmat)), 'ascend');
    fprintf('    done in %.1f s\n', toc(t0));

    fprintf('  smallest %d eigs of L^{-1} A L^{-T} (via inverse handle)...\n', k);
    Lt   = L';
    dA   = decomposition(A, 'chol');
    Tinv = @(x) Lt * (dA \ (L * x));
    t0   = tic;
    [~, Dmat] = eigs(Tinv, n, k, 'smallestabs', opts);
    D_M = sort(real(diag(Dmat)), 'ascend');
    fprintf('    done in %.1f s\n', toc(t0));

    save(cachePath, 'D_A', 'D_M', 'n', 'h0', 'k', '-v7');
    fprintf('  saved %s\n', cachePath);
end

function [D_A, D_M, n] = load_or_compute_eigs_refine_large(cacheDir, h0, contrast, t_snap, dt, Tmax, mesh_method, k)
%LOAD_OR_COMPUTE_EIGS_REFINE_LARGE  Largest k eigs of A and Tsym at h0.
    cachePath = fullfile(cacheDir, ...
        sprintf('eigs_refine_large_h0_%s_k%d.mat', tag(h0), k));
    if isfile(cachePath)
        S = load(cachePath, 'D_A', 'D_M', 'n');
        D_A = S.D_A; D_M = S.D_M; n = S.n;
        fprintf('h0=%.4g  cached %s (n=%d)\n', h0, cachePath, n);
        return;
    end
    fprintf('\n--- h0=%.4g : largest-k eigs ---\n', h0);
    [A, L] = build_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method);
    n = size(A, 1);

    fprintf('  largest %d eigs of A...\n', k);
    opts = struct('Tolerance', 1e-10, 'MaxIterations', 5000);
    t0   = tic;
    [~, Dmat] = eigs(A, k, 'largestabs', opts);
    D_A = sort(real(diag(Dmat)), 'descend');
    fprintf('    done in %.1f s\n', toc(t0));

    fprintf('  largest %d eigs of L^{-1} A L^{-T} (forward handle)...\n', k);
    Lt   = L';
    Tfun = @(x) L \ (A * (Lt \ x));
    opts_T = struct('Tolerance', 1e-10, 'MaxIterations', 5000, ...
                    'IsFunctionSymmetric', true);
    t0   = tic;
    [~, Dmat] = eigs(Tfun, n, k, 'largestabs', opts_T);
    D_M = sort(real(diag(Dmat)), 'descend');
    fprintf('    done in %.1f s\n', toc(t0));

    save(cachePath, 'D_A', 'D_M', 'n', 'h0', 'k', '-v7');
    fprintf('  saved %s\n', cachePath);
end

function D_k = load_D_from_cache(cachePath, k)
%LOAD_D_FROM_CACHE  Smallest k eigenvalues from a saved cache (D ascending).
    S = load(cachePath, 'D');
    D = S.D(:);
    D_k = D(1:k);
end

function D_k = load_D_from_cache_desc(cachePath, k)
%LOAD_D_FROM_CACHE_DESC  Largest k eigenvalues from a saved cache (D descending).
    S = load(cachePath, 'D');
    D = S.D(:);
    D_k = D(1:k);
end

function s = tag(h0)
%TAG  File-safe string for an h0 value (no dots).
    s = strrep(sprintf('%.4g', h0), '.', 'p');
end

function [P0, ncols] = build_P0(kind, Pt_A, nc_A, Pt_T, nc_T, n, m)
%BUILD_P0  Construct a starting block of one of the supported kinds.
    switch kind
        case 'gaussian'
            P0    = randn(n, m);
            ncols = m;
        case 'tent_A'
            P0    = full(Pt_A);
            ncols = nc_A;
        case 'sketched_tent_A'
            G  = randn(nc_A, m);
            P0 = Pt_A * G;
            ncols = m;
        case 'tent_T'
            P0    = full(Pt_T);
            ncols = nc_T;
        case 'sketched_tent_T'
            G  = randn(nc_T, m);
            P0 = Pt_T * G;
            ncols = m;
        otherwise
            error('build_P0: unknown kind %s', kind);
    end
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

function T_sparse = build_Tsym_sparse(L, Lt, A, drop_rel_tol)
%BUILD_TSYM_SPARSE  Materialize Tsym = L^{-1} A L^{-T}, symmetrize, drop near-zeros.
    Tdense = (L \ full(A)) / Lt;
    Tdense = 0.5 * (Tdense + Tdense.');
    cutoff = drop_rel_tol * max(abs(Tdense(:)));
    Tdense(abs(Tdense) < cutoff) = 0;
    T_sparse = sparse(Tdense);
end

function info = run_one(Zfun, P0, poly, i_deg, bnds, V_true)
%RUN_ONE  Apply one swept-degree polynomial to P0, then measure capture.
%   Handles the fixed-degree families 'power_iz' and 'chebyshev'.  The
%   balanced Chebyshev family is auto-degree and lives in run_balanced_opt.
%   bnds: struct with fields lam_first (D(1)), lam_cut (D(k+1) stop-band edge),
%         lam_max (top of spectrum).
    info = new_capture_info();
    n  = size(P0, 1);
    t0 = tic;
    try
        switch poly
            case 'chebyshev'
                % Original high-pass T_i filter; reject band [lam_cut, lam_max].
                Y = src.precond.chebyshev_apply(Zfun, P0, i_deg, ...
                                                bnds.lam_cut, bnds.lam_max);
                Q = orth(Y);
            case 'power_iz'
                Dinv = (1 / bnds.lam_max) * ones(n, 1);
                Q = src.precond.min_subspace_iter(Zfun, P0, i_deg, ...
                                                  Dinv, 1.0, false);
            otherwise
                error('run_one: unknown poly %s', poly);
        end
        info = fill_capture_info(info, V_true, Q);
    catch ME
        info.err = regexprep(ME.message, '\n.*', '');
        warning('run_subspace_capture:run_one_failed', ...
                'poly=%s i=%d failed: %s', poly, i_deg, info.err);
    end
    info.time_seconds = toc(t0);
end

function [info, dstar] = run_balanced_opt(Zfun, P0, bnds, V_true)
%RUN_BALANCED_OPT  Paper-optimal balanced Chebyshev filter (Saad, 1512.08135).
%   Auto-selects the degree (smallest degree whose filter value at the band
%   edge drops below phi) with Lanczos sigma-damping, left-end centered, for
%   the wanted band [lam_first, lam_cut].  Returns the capture info and the
%   selected degree dstar = filt.k (one optimal operating point, not a sweep).
    info  = new_capture_info();
    dstar = NaN;
    t0    = tic;
    try
        % Name-value opts (the arguments block rejects a packed struct); omit
        % 'degree' so the function selects the optimal degree automatically.
        filt = make_balanced_cheb_filter(bnds.lam_first, bnds.lam_max, ...
                    [bnds.lam_first, bnds.lam_cut], ...
                    'mode', 'left', 'damping', bnds.damping, 'phi', bnds.phi);
        Y    = apply_cheb_filter(Zfun, P0, filt);
        Q    = orth(Y);
        info = fill_capture_info(info, V_true, Q);
        dstar = filt.k;
    catch ME
        info.err = regexprep(ME.message, '\n.*', '');
        warning('run_subspace_capture:run_balanced_opt_failed', ...
                'chebyshev_balanced failed: %s', info.err);
    end
    info.time_seconds = toc(t0);
end

function info = new_capture_info()
%NEW_CAPTURE_INFO  Empty result struct shared by run_one / run_balanced_opt.
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
    fprintf(fid, ['Z,P0_kind,P0_ncols,polynomial,degree,', ...
                  'max_residual,mean_residual,n_res_below_1pct,', ...
                  'n_res_below_0p1pct,capture_frac_1pct,max_principal_angle,', ...
                  'time_seconds\n']);
    for i = 1:numel(rows)
        r = rows(i);
        fprintf(fid, '%s,%s,%d,%s,%d,%g,%g,%g,%g,%g,%g,%g\n', ...
                r.Z, r.P0_kind, r.P0_ncols, r.poly, r.degree, ...
                r.max_residual, r.mean_residual, ...
                r.n_res_below_1pct, r.n_res_below_0p1pct, ...
                r.capture_frac_1pct, r.max_principal_angle, ...
                r.time_seconds);
    end
    fclose(fid);
end

function f = make_latitude_banding_contrast(Tmax, contrast)
%MAKE_LATITUDE_BANDING_CONTRAST  Latitude-banding kappa with adjustable contrast.
%   Kept LOCAL (not in +src): kappa factories are experiment-specific, change
%   constantly, and are not persistent across geometry.
%
%   F = MAKE_LATITUDE_BANDING_CONTRAST(TMAX, CONTRAST) returns a handle
%   @(x,y,z,t) producing three drifting latitude bands.  contrast =
%   kappa_max/kappa_min is realized about the geometric mean
%   (kmin = 1/sqrt(contrast), kmax = sqrt(contrast)) so the typical
%   diffusivity stays O(1) across a contrast sweep.
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
    % Axisymmetric: kappa depends only on colatitude (z); x, y carry shape only.
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
%% Rendering: capture plots (4 PDFs + 1 aggregate PNG)
%% =========================================================================
function specs = panel_specs()
%PANEL_SPECS  The four (metric, Z) capture panels, shared by PDF + aggregate.
%   Field `tag` is the file stem for the per-panel PDFs.
    specs = struct( ...
        'metric',     {'max_residual',  'capture_frac_1pct',  'max_residual',     'capture_frac_1pct'}, ...
        'Z',          {'A',             'A',                  'Tsym',             'Tsym'}, ...
        'Zlabel',     {'A',             'A',                  'L^{-1} A L^{-T}',  'L^{-1} A L^{-T}'}, ...
        'yscale',     {'log',           'linear',             'log',              'linear'}, ...
        'ylabel',     {'max residual',  'capture fraction (res < 1%)', ...
                       'max residual',  'capture fraction (res < 1%)'}, ...
        'titlebit',   {'max residual',  'capture fraction',   'max residual',     'capture fraction'}, ...
        'legend_loc', {'southwest',     'northwest',          'south',            'south'}, ...
        'tag',        {'max_residual_A', 'capture_fraction_A', ...
                       'max_residual_Tsym', 'capture_fraction_Tsym'});
end

function make_ablation_paper(rows, out_dir)
%MAKE_ABLATION_PAPER  Paper-legible polynomial-filter capture plots.
%
%   MAKE_ABLATION_PAPER(ROWS, OUT_DIR) emits four vector PDFs into OUT_DIR:
%     max_residual_A.pdf       — max per-vector residual vs degree, Z = A
%     capture_fraction_A.pdf   — fraction of vectors with res<1%, Z = A
%     max_residual_Tsym.pdf    — same, Z = L^{-1} A L^{-T}
%     capture_fraction_Tsym.pdf
%   Each panel overlays power_iz / chebyshev / chebyshev_balanced for the
%   gaussian and sketched-tent starting blocks.

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    specs = panel_specs();
    for ip = 1:numel(specs)
        spec    = specs(ip);
        outfile = fullfile(out_dir, [spec.tag '.pdf']);
        fig = figure('Visible', 'off', 'Units', 'inches', ...
                     'Position', [0 0 5.0 3.2], 'Color', 'w');
        draw_panel(axes(fig), rows, spec);
        exportgraphics(fig, outfile, 'ContentType', 'vector');
        close(fig);
        fprintf('Wrote %s\n', outfile);
    end
end

function make_ablation_aggregate_png(rows, out_dir)
%MAKE_ABLATION_AGGREGATE_PNG  All four capture panels on one 2x2 PNG.
%   Rows = Z {A, Tsym}; columns = metric {max residual, capture fraction}.
%   Large canvas so the (up to 9) per-tile legends stay legible.
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    % Order the 4 specs into the 2x2 grid (row-major: A row, then Tsym row).
    specs = panel_specs();
    order = [find_spec(specs,'A','max_residual'),    find_spec(specs,'A','capture_frac_1pct'), ...
             find_spec(specs,'Tsym','max_residual'), find_spec(specs,'Tsym','capture_frac_1pct')];

    fig = figure('Visible', 'off', 'Units', 'inches', ...
                 'Position', [0 0 12 9], 'Color', 'w');
    tl = tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for ii = 1:4
        draw_panel(nexttile(tl), rows, specs(order(ii)));
    end
    title(tl, 'Polynomial-filter subspace capture', ...
          'FontWeight', 'bold', 'FontSize', 13);

    outfile = fullfile(out_dir, 'aggregate_2x2.png');
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
    fprintf('Wrote %s\n', outfile);
end

function idx = find_spec(specs, Z, metric)
%FIND_SPEC  Index of the spec with the given Z and metric.
    idx = find(strcmp({specs.Z}, Z) & strcmp({specs.metric}, metric), 1);
end

function draw_panel(ax, rows, spec)
%DRAW_PANEL  Draw one (metric, Z) capture panel into axes AX.
%   Shared by the per-panel PDFs and the 2x2 aggregate PNG.
    is_Tsym = strcmp(spec.Z, 'Tsym');
    if is_Tsym
        P0_kinds = {'gaussian', 'sketched_tent_A', 'sketched_tent_T'};
    else
        P0_kinds = {'gaussian', 'sketched_tent_A'};
    end
    poly_kinds = {'power_iz', 'chebyshev', 'chebyshev_balanced'};

    series = series_spec(P0_kinds, poly_kinds);
    hold(ax, 'on');

    legs    = cell(1, numel(series));
    handles = gobjects(1, numel(series));
    keep    = false(1, numel(series));
    yall    = [];
    xall    = [];

    for si = 1:numel(series)
        s = series(si);
        mask = strcmp({rows.Z},       spec.Z)    & ...
               strcmp({rows.P0_kind}, s.P0_kind) & ...
               strcmp({rows.poly},    s.poly);
        sub  = rows(mask);
        if isempty(sub), continue; end

        [degs, ord] = sort([sub.degree]);
        yvals = [sub.(spec.metric)];
        yvals = yvals(ord);

        if isscalar(degs)
            % single auto-degree optimal point (balanced) -> big filled,
            % black-edged marker so it pops even at y=0 / among same-color curves.
            handles(si) = plot(ax, degs, yvals, s.mark, ...
                               'Color', s.color, 'MarkerFaceColor', s.color, ...
                               'MarkerEdgeColor', 'k', 'MarkerSize', 11, ...
                               'LineWidth', 1.0);
        else
            handles(si) = plot(ax, degs, yvals, [s.style s.mark], ...
                               'Color', s.color, 'MarkerSize', 4.5, ...
                               'LineWidth', 1.0);
        end
        legs{si} = s.label;
        keep(si) = true;
        yall = [yall, yvals]; %#ok<AGROW>
        xall = [xall, degs];  %#ok<AGROW>
    end
    hold(ax, 'off');

    set(ax, 'XScale', 'linear', 'YScale', spec.yscale, 'Box', 'on', ...
            'LineWidth', 0.6, 'FontSize', 9, ...
            'XGrid', 'off', 'YGrid', 'off');

    % X ticks: draw a tick at every sweep degree, but suppress labels at the
    % crowded low end so they don't run together visually.
    if ~isempty(xall)
        xt = unique(xall);
        xlim(ax, [min(xt) - 1, max(xt) + 1]);
        xt_labels = arrayfun(@(v) sprintf('%g', v), xt, 'UniformOutput', false);
        xt_labels(xt < 5 & xt > 0) = {''};
        set(ax, 'XTick', xt, 'XTickLabel', xt_labels);
    end

    if strcmp(spec.yscale, 'linear') && contains(spec.ylabel, 'capture')
        ylim(ax, [-0.02, 1.05]);
        set(ax, 'YTick', 0:0.2:1);
    elseif strcmp(spec.yscale, 'log') && ~isempty(yall)
        yall_pos = yall(yall > 0);
        if ~isempty(yall_pos)
            ymin = min(yall_pos);
            ymax = max(yall_pos);
            ylim(ax, [ymin * 0.5, ymax * 1.5]);
        end
    end

    xlabel(ax, 'polynomial degree i', ...
           'FontSize', 10, 'FontWeight', 'bold');
    ylabel(ax, spec.ylabel, ...
           'FontSize', 10, 'FontWeight', 'bold');
    title(ax, sprintf('%s  (Z = %s)', spec.titlebit, spec.Zlabel), ...
          'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'tex');

    lgd = legend(ax, handles(keep), legs(keep), ...
                 'Location',    spec.legend_loc, ...
                 'Box',         'on', ...
                 'EdgeColor',   [0.65 0.65 0.65], ...
                 'Color',       'white', ...
                 'FontSize',    6.5, ...
                 'NumColumns',  2, ...
                 'Interpreter', 'none');
    lgd.ItemTokenSize = [9, 5];
end

function series = series_spec(P0_kinds, poly_kinds)
%SERIES_SPEC  (P0_kind, poly) cartesian product with consistent visuals.
%   Color encodes P0_kind, line style + marker encode polynomial family.
    P0_color = struct( ...
        'gaussian',        [0.00 0.00 0.00], ...
        'tent_A',          [0.20 0.40 0.80], ...
        'sketched_tent_A', [0.85 0.33 0.10], ...
        'tent_T',          [0.20 0.70 0.30], ...
        'sketched_tent_T', [0.55 0.15 0.65]);
    poly_style  = struct('power_iz', '--', 'chebyshev', '-',  'chebyshev_balanced', ':');
    poly_marker = struct('power_iz', 'o',  'chebyshev', 's',  'chebyshev_balanced', 'p');

    n = numel(P0_kinds) * numel(poly_kinds);
    series = repmat(struct('P0_kind', '', 'poly', '', 'label', '', ...
                           'style', '', 'mark', '', 'color', [0 0 0]), 1, n);
    idx = 0;
    for ik = 1:numel(P0_kinds)
        for ip = 1:numel(poly_kinds)
            idx = idx + 1;
            P0k  = P0_kinds{ik};
            plyk = poly_kinds{ip};
            series(idx).P0_kind = P0k;
            series(idx).poly    = plyk;
            series(idx).label   = sprintf('%s / %s', P0k, plyk);
            series(idx).style   = poly_style.(plyk);
            series(idx).mark    = poly_marker.(plyk);
            series(idx).color   = P0_color.(P0k);
        end
    end
end

%% =========================================================================
%% Rendering: spectrum / spy diagnostics (5 PDFs)
%% =========================================================================
function plot_spectrum_and_spy(sd, out_dir)
%PLOT_SPECTRUM_AND_SPY  Five spectrum / spy figures from precomputed spectra.
%
%   PLOT_SPECTRUM_AND_SPY(SD, OUT_DIR) renders five vector PDFs into OUT_DIR.
%   All eigen-computation and snapshot assembly happen above; this only draws.
%
%   Outputs:
%     eig_spectrum.pdf                    — smallest k eigenvalues of A and
%                                           M = L^{-1} A L^{-T} (baseline h0).
%     spy.pdf                             — spy(A) | spy(L = ichol(A,'nofill')).
%     eig_spectrum_mesh_refine.pdf        — smallest-k overlay across h0 sweep.
%     eig_spectrum_large.pdf              — largest k eigenvalues (baseline h0).
%     eig_spectrum_large_mesh_refine.pdf  — largest-k overlay across h0 sweep.
%
%   SD fields: h0_base, n_base, k, A, L, D_A_base, D_M_base, D_A_lg, D_M_lg,
%   h0_refine, n_all, D_A_all, D_M_all, D_A_lg_all, D_M_lg_all.

    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    figPath1 = fullfile(out_dir, 'eig_spectrum.pdf');
    plot_eigs_overlay(sd.D_A_base, sd.D_M_base, sd.h0_base, sd.n_base, sd.k, ...
                      'smallest', figPath1);
    fprintf('Saved %s\n', figPath1);

    figPath2 = fullfile(out_dir, 'spy.pdf');
    plot_spy_overlay(sd.A, sd.L, sd.h0_base, figPath2);
    fprintf('Saved %s\n', figPath2);

    figPath3 = fullfile(out_dir, 'eig_spectrum_mesh_refine.pdf');
    plot_mesh_refine_overlay(sd.h0_refine, sd.n_all, sd.D_A_all, sd.D_M_all, ...
                             sd.k, 'smallest', figPath3);
    fprintf('Saved %s\n', figPath3);

    figPath4 = fullfile(out_dir, 'eig_spectrum_large.pdf');
    plot_eigs_overlay(sd.D_A_lg, sd.D_M_lg, sd.h0_base, sd.n_base, sd.k, ...
                      'largest', figPath4);
    fprintf('Saved %s\n', figPath4);

    figPath5 = fullfile(out_dir, 'eig_spectrum_large_mesh_refine.pdf');
    plot_mesh_refine_overlay(sd.h0_refine, sd.n_all, sd.D_A_lg_all, sd.D_M_lg_all, ...
                             sd.k, 'largest', figPath5);
    fprintf('Saved %s\n', figPath5);
end

function plot_eigs_overlay(D_A, D_M, h0, n, k, mode, figPath)
%PLOT_EIGS_OVERLAY  Single-mesh A vs M spectrum overlay.
%   mode = 'smallest' : D vectors ascending, x-axis reversed (k -> 1).
%   mode = 'largest'  : D vectors descending, natural x-axis (1 -> k).
    f  = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1100 700]);
    ax = axes(f); hold(ax, 'on'); grid(ax, 'on');

    D_A_plot = D_A(:); D_A_plot(D_A_plot <= 0) = NaN;
    D_M_plot = D_M(:); D_M_plot(D_M_plot <= 0) = NaN;

    hA = semilogy(ax, 1:numel(D_A_plot), D_A_plot, '-', ...
                  'LineWidth', 1.6, 'Color', [0.85 0.40 0.32]);
    hM = semilogy(ax, 1:numel(D_M_plot), D_M_plot, '-', ...
                  'LineWidth', 1.6, 'Color', [0.20 0.45 0.70]);

    if strcmp(mode, 'smallest')
        xdir = 'reverse';
        x_label_txt  = sprintf('index i (%d \\rightarrow 1)', k);
        title_txt    = sprintf('ball\\_surface  h0=%.4g  n=%d : smallest %d eigenvalues', ...
                               h0, n, k);
        end_label    = 'min';
    else
        xdir = 'normal';
        x_label_txt  = sprintf('index i (1 \\rightarrow %d)', k);
        title_txt    = sprintf('ball\\_surface  h0=%.4g  n=%d : largest %d eigenvalues', ...
                               h0, n, k);
        end_label    = 'max';
    end
    set(ax, 'YScale', 'log', 'XDir', xdir, 'FontSize', 12);
    xlabel(ax, x_label_txt, 'FontSize', 12);
    ylabel(ax, '\lambda_i  (log scale)', 'FontSize', 12);
    title(ax, title_txt, 'FontSize', 13);

    sub_lines = {};
    pA = D_A(D_A > 0);
    if ~isempty(pA)
        sub_lines{end+1} = sprintf('A: \\lambda_{%s}=%.3e,  \\lambda_{%d}=%.3e', ...
                                   end_label, pA(1), k, D_A(end));
    end
    pM = D_M(D_M > 0);
    if ~isempty(pM)
        sub_lines{end+1} = sprintf('M: \\lambda_{%s}=%.3e,  \\lambda_{%d}=%.3e', ...
                                   end_label, pM(1), k, D_M(end));
    end
    if ~isempty(sub_lines), subtitle(ax, strjoin(sub_lines, '   |   ')); end

    legend(ax, [hA, hM], {'A', 'M = L^{-1} A L^{-T}'}, ...
           'Location', 'best', 'FontSize', 12, 'Box', 'on');
    hold(ax, 'off');
    exportgraphics(f, figPath, 'ContentType', 'vector');
    close(f);
end

function plot_spy_overlay(A, L, h0, figPath)
%PLOT_SPY_OVERLAY  spy(A) | spy(L) side-by-side.
    f  = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1200 550]);
    tl = tiledlayout(f, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile(tl); spy(A);
    title(sprintf('A   (n=%d, nnz=%d)', size(A, 1), nnz(A)), ...
          'Interpreter', 'none');
    nexttile(tl); spy(L);
    title(sprintf('L = ichol(A, ''nofill'')   (nnz(L)=%d, ratio=%.3f)', ...
          nnz(L), nnz(L) / max(nnz(A), 1)), 'Interpreter', 'none');
    sgtitle(tl, sprintf('ball\\_surface  h0=%.4g', h0), 'FontWeight', 'bold');
    exportgraphics(f, figPath, 'ContentType', 'vector');
    close(f);
end

function plot_mesh_refine_overlay(h0_vals, n_vals, D_A_cell, D_M_cell, k, mode, figPath)
%PLOT_MESH_REFINE_OVERLAY  A and M spectra overlaid across h0 values.
%   Warm shades for A (darker = finer mesh), cool shades for M.
    nH = numel(h0_vals);

    % Sort coarsest -> finest by h0 (descending so finest is last/darkest).
    [h0_sorted, ord] = sort(h0_vals, 'descend');
    n_sorted = n_vals(ord);
    D_A_cell = D_A_cell(ord);
    D_M_cell = D_M_cell(ord);

    % Light -> dark interpolation for each family.
    A_light = [0.97 0.78 0.74];   A_dark = [0.55 0.10 0.08];
    M_light = [0.74 0.84 0.95];   M_dark = [0.06 0.22 0.50];

    f  = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1200 750]);
    ax = axes(f); hold(ax, 'on'); grid(ax, 'on');

    legH = gobjects(2 * nH, 1);
    legL = cell(2 * nH, 1);

    for ih = 1:nH
        if nH == 1
            tA = 1; tM = 1;
        else
            tA = (ih - 1) / (nH - 1);
            tM = (ih - 1) / (nH - 1);
        end
        cA = (1 - tA) * A_light + tA * A_dark;
        cM = (1 - tM) * M_light + tM * M_dark;

        DA = D_A_cell{ih}(:); DA(DA <= 0) = NaN;
        DM = D_M_cell{ih}(:); DM(DM <= 0) = NaN;

        hA = semilogy(ax, 1:numel(DA), DA, '-', 'LineWidth', 1.6, 'Color', cA);
        hM = semilogy(ax, 1:numel(DM), DM, '-', 'LineWidth', 1.6, 'Color', cM);

        legH(2*ih - 1) = hA;
        legH(2*ih)     = hM;
        legL{2*ih - 1} = sprintf('A   (h0=%.4g, n=%d)', h0_sorted(ih), n_sorted(ih));
        legL{2*ih}     = sprintf('M   (h0=%.4g, n=%d)', h0_sorted(ih), n_sorted(ih));
    end

    if strcmp(mode, 'smallest')
        xdir = 'reverse';
        x_label_txt = sprintf('index i (%d \\rightarrow 1)', k);
        title_txt   = sprintf('ball\\_surface : smallest %d eigenvalues, mesh refinement', k);
    else
        xdir = 'normal';
        x_label_txt = sprintf('index i (1 \\rightarrow %d)', k);
        title_txt   = sprintf('ball\\_surface : largest %d eigenvalues, mesh refinement', k);
    end
    set(ax, 'YScale', 'log', 'XDir', xdir, 'FontSize', 12);
    xlabel(ax, x_label_txt, 'FontSize', 12);
    ylabel(ax, '\lambda_i  (log scale)', 'FontSize', 12);
    title(ax, title_txt, 'FontSize', 13);
    subtitle(ax, sprintf('h0 \\in \\{%s\\}   (darker = finer mesh)', ...
             strjoin(arrayfun(@(x) sprintf('%.4g', x), h0_sorted, ...
                              'UniformOutput', false), ', ')));

    legend(ax, legH, legL, 'Location', 'best', 'FontSize', 11, 'Box', 'on', ...
           'NumColumns', 2);
    hold(ax, 'off');
    exportgraphics(f, figPath, 'ContentType', 'vector');
    close(f);
end
