function [A, L, msh, b, meta] = amg_bench_snapshot(h0, contrast, t_snap, dt, ...
                                                   Tmax, mesh_method, ...
                                                   Kmodes, sigma, seed)
%AMG_BENCH_SNAPSHOT  The single definition of the amg_subspace test problem.
%
%   [A, L, msh] = amg_bench_snapshot(h0, contrast, t_snap, dt, Tmax, mesh_method)
%   builds the closed-sphere surface-FEM heat-step operator
%       A = D_II + dt * K_II(kappa(.,t_snap))          (SPD)
%   with the latitude-banding diffusivity of the given contrast, together with
%   its incomplete Cholesky factor L = ichol(A,'nofill') (diagcomp fallback).
%
%   [A, L, msh, b, meta] = amg_bench_snapshot(..., Kmodes, sigma, seed)
%   additionally returns the KL cosine-mode right-hand side used by the
%   downstream (P)CG solves,
%       b = sigma * sqrt(dt) * D_II * (Phi * z),   z ~ N(0, I_Kmodes),
%   matching subspace_capture/run_krylov_capture.m and
%   subspace_capture/run_inverse_subspace_iter.m so iteration counts are
%   directly comparable across those studies.  Computing b costs a mesh
%   evaluation of Kmodes cosine modes, so callers that only need [A, L, msh]
%   should request three outputs and skip it.
%
%   This function exists so the three amg_subspace drivers share ONE
%   definition of the test problem; the helpers below were previously
%   duplicated as file-local subfunctions of each driver.  The assembled A
%   must stay bit-identical to that earlier code, since the committed
%   eigenvector caches (output/cache/eigsA_h0.05_k200.mat and friends) are
%   keyed only by (h0, k).  test_amg_deflation_arms.m asserts this.
%
%   Inputs
%     h0          : target mesh size passed to src.discretization.build_sphere_mesh
%     contrast    : kappa_max/kappa_min ratio of the banding field (>= 1)
%     t_snap      : time level at which kappa is frozen
%     dt          : time-step multiplying the stiffness block
%     Tmax        : horizon that sets the kappa band-drift frequencies
%     mesh_method : mesh generator, e.g. 'pdetoolbox'
%     Kmodes      : number of KL cosine modes in b        (default 50)
%     sigma       : RHS scale; irrelevant to Krylov iteration counts, since
%                   (P)CG normalizes by ||b||             (default 1)
%     seed        : rng seed for the mode coefficients z  (default 1)
%
%   Outputs
%     A    : numIN-by-numIN SPD sparse operator
%     L    : lower incomplete Cholesky factor of A
%     msh  : mesh struct from src.discretization.build_sphere_mesh
%     b    : numIN-by-1 right-hand side (empty if fewer than 4 outputs asked)
%     meta : struct recording every input plus n, nnz(A), nnz(L), norm(b)
%
%   See also RUN_AMG_DEFLATION_VS_PRECOND, RUN_AMG_SUBSPACE_CAPTURE,
%   PRECOND_SPECTRUM.

    if nargin < 7 || isempty(Kmodes), Kmodes = 50; end
    if nargin < 8 || isempty(sigma),  sigma  = 1;  end
    if nargin < 9 || isempty(seed),   seed   = 1;  end

    validateattributes(h0,       {'numeric'}, {'scalar', 'positive', 'finite'});
    validateattributes(contrast, {'numeric'}, {'scalar', 'finite', '>=', 1});
    validateattributes(dt,       {'numeric'}, {'scalar', 'positive', 'finite'});
    validateattributes(Tmax,     {'numeric'}, {'scalar', 'positive', 'finite'});
    validateattributes(Kmodes,   {'numeric'}, {'scalar', 'integer', 'positive'});

    msh      = src.discretization.build_sphere_mesh(h0, false, mesh_method);
    kappaFun = make_latitude_banding_contrast(Tmax, contrast);
    A        = assemble_snapshot_A(msh, kappaFun, dt, t_snap);
    L        = ichol_with_fallback(A);

    b = [];
    if nargout >= 4
        % KL-noise RHS.  bbox is the unit sphere's x,y extent; the cosine
        % modes are evaluated at the mesh vertices and mass-weighted, exactly
        % as the time-stepping solver forms its forcing.
        bbox  = [-1 1 -1 1];
        kvec  = generate_kvec(Kmodes);
        Phi   = src.forcing.eval_cosine_modes(msh.p, kvec, bbox);
        rng(seed);
        z     = randn(Kmodes, 1);
        b     = sigma * sqrt(dt) * (msh.D_II * (Phi * z));
    end

    if nargout >= 5
        meta = struct('h0', h0, 'contrast', contrast, 't_snap', t_snap, ...
                      'dt', dt, 'Tmax', Tmax, 'mesh_method', mesh_method, ...
                      'Kmodes', Kmodes, 'sigma', sigma, 'seed', seed, ...
                      'n', msh.numIN, 'nnzA', nnz(A), 'nnzL', nnz(L), ...
                      'norm_b', norm(b));
    end
end

%% =========================================================================
%% Local helpers
%% =========================================================================
function A = assemble_snapshot_A(msh, kappaFun, dt, tcur)
%ASSEMBLE_SNAPSHOT_A  Build A = D_II + dt*K_II at one time level (closed sphere).
%   Copied verbatim from run_amg_subspace_capture.m (pre-extraction).
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
%   Copied verbatim from run_amg_subspace_capture.m (pre-extraction).
    try
        L = ichol(A, struct('type', 'nofill'));
    catch
        alpha    = max(sum(abs(A), 2) ./ diag(A)) - 2;
        diagcomp = max(alpha, 0);
        L = ichol(A, struct('type', 'nofill', 'diagcomp', diagcomp));
    end
end

function f = make_latitude_banding_contrast(Tmax, contrast)
%MAKE_LATITUDE_BANDING_CONTRAST  Latitude-banding kappa with adjustable contrast.
%   Kept LOCAL (not in +src): kappa factories are experiment-specific.
%   Copied verbatim from run_amg_subspace_capture.m (pre-extraction).
    if nargin < 2 || isempty(contrast), contrast = 60; end
    if contrast < 1
        error('amg_bench_snapshot:badContrast', ...
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
%LATITUDE_BANDING_EVAL  Three drifting tanh bands in colatitude.
%   Copied verbatim from run_amg_subspace_capture.m (pre-extraction).
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
%   Copied verbatim from subspace_capture/run_inverse_subspace_iter.m so the
%   RHS matches the sibling studies' solver.
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
