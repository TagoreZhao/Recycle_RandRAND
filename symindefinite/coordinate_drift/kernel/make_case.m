function cs = make_case(family, idx, opts)
%MAKE_CASE  One interface, two preconditioner families.
%
%   cs = MAKE_CASE(FAMILY, IDX)
%   cs = MAKE_CASE(FAMILY, IDX, OPTS)
%
%   Returns the step-IDX system together with its chart, so that every
%   experiment in this study runs the SAME code path for
%
%     'ildl'   symmetric indefinite.  A = K_n, the immersed-rotor KKT matrix at
%              step IDX (build_stokes_sequence / seq_K); C = S^-1 P' L |D|^{1/2}
%              from src.precond.make_ildl_precond.
%
%     'ichol'  SPD.  A = K_s + sigma2(IDX)*I, a sparsified RBF kernel-ridge
%              system from the GP_train/pumadyn32nm data; C = L from
%              ichol with a FIXED sparsity pattern.
%
%   Controlling both families through one interface is what makes the
%   SPD-vs-indefinite comparison a controlled experiment rather than two
%   separate stories: the only thing that differs between the two columns of
%   every table is (A_n, C_n).
%
%   Why the kernel is sparsified: ichol('nofill') on a dense pattern is an EXACT
%   Cholesky, which gives M = A, Ahat = I and a degenerate study.  Thresholding
%   the RBF kernel to ~1% density makes the factorization genuinely incomplete
%   while keeping the pattern FIXED along the sigma2 sweep -- exactly the
%   hypothesis of Thm 3.1.  opts.ichol_type = 'ict' switches to the benchmark's
%   value-dependent drop pattern for contrast.
%
%   OPTS (all optional):
%     .case_name  Stokes motion, default 'bar_rotating' ('disk_static' = the
%                 null control, where K and hence C never move)
%     .h0         Stokes mesh size, default 0.15 (fast); 0.05 = benchmark scale
%     .nsteps     Stokes steps to build, default 8
%     .n_gp       SPD problem size, default 800
%     .density    target kernel density after thresholding, default 0.02
%     .sigma2     the SPD ladder, default logspace(-4, 0, 8) offset to keep SPD
%     .ichol_type 'nofill' (default) | 'ict'
%     .droptol    ichol ict droptol, default 1e-3
%     .want_M     also return the explicit M = C*C' (default true)
%
%   cs fields:
%     .A .C .M                  system, chart factor, metric
%     .applyCinv .applyCtinv    the split solves (Ahat = C^-1 A C^-T)
%     .P                        the ILDL struct ([] for 'ichol')
%     .n .family .idx .label .param
%
%   The heavy base data (the Stokes sequence, the kernel matrix) is built once
%   and memoized, so a loop over IDX pays for it only on the first call.
%
%   See also: add_paths, pencil_subspace, gauge_split.

    if nargin < 3 || isempty(opts), opts = struct(); end

    switch lower(family)
        case 'ildl',  cs = case_ildl(idx, opts);
        case 'ichol', cs = case_ichol(idx, opts);
        otherwise
            error('make_case:family', 'unknown family "%s"', family);
    end

    cs.family = lower(family);
    cs.idx    = idx;
    cs.n      = size(cs.A, 1);
    if getdef(opts, 'want_M', true)
        cs.M = cs.C * cs.C';
        cs.M = (cs.M + cs.M') / 2;
    else
        cs.M = [];
    end
end

%==========================================================================
function cs = case_ildl(idx, opts)
    S = stokes_seq(opts);
    A = seq_K(S, idx);
    P = src.precond.make_ildl_precond(A, struct('mode', 'nofill'));
    C = ildl_coordinate_map(P);

    cs = struct();
    cs.A          = A;
    cs.C          = C;
    cs.P          = P;
    cs.applyCinv  = P.applyCinv;
    cs.applyCtinv = P.applyCtinv;
    cs.param      = idx;                       % the "time" coordinate
    cs.label      = sprintf('ildl step %d', idx);
    cs.nC         = S.nC;
end

function cs = case_ichol(idx, opts)
    G      = gp_kernel(opts);
    sig2   = G.sigma2(idx);
    A      = G.Ks + sig2 * speye(G.n);
    itype  = getdef(opts, 'ichol_type', 'nofill');
    if strcmp(itype, 'nofill')
        L = ichol(A, struct('type', 'nofill'));
    else
        L = ichol(A, struct('type', 'ict', ...
                            'droptol', getdef(opts, 'droptol', 1e-3), ...
                            'michol', 'on'));
    end

    cs = struct();
    cs.A          = A;
    cs.C          = L;
    cs.P          = [];
    cs.applyCinv  = @(r) L  \ r;
    cs.applyCtinv = @(y) L' \ y;
    cs.param      = sig2;
    cs.label      = sprintf('ichol sigma2=%.3g', sig2);
    cs.nC         = 0;
end

%==========================================================================
function S = stokes_seq(opts)
%STOKES_SEQ  Memoized immersed-rotor KKT sequence.
    persistent CACHE
    key = sprintf('%s|%g|%d', getdef(opts, 'case_name', 'bar_rotating'), ...
                              getdef(opts, 'h0', 0.15), ...
                              getdef(opts, 'nsteps', 8));
    if ~isempty(CACHE) && isfield(CACHE, 'key') && strcmp(CACHE.key, key)
        S = CACHE.S;  return;
    end
    S = build_stokes_sequence(struct( ...
            'case_name', getdef(opts, 'case_name', 'bar_rotating'), ...
            'h0',        getdef(opts, 'h0',        0.15), ...
            'nsteps',    getdef(opts, 'nsteps',    8), ...
            'quiet',     true));
    CACHE = struct('key', key, 'S', S);
end

function G = gp_kernel(opts)
%GP_KERNEL  Memoized sparsified RBF kernel-ridge data.
    persistent CACHE
    n_gp    = getdef(opts, 'n_gp',    800);
    density = getdef(opts, 'density', 0.02);
    key     = sprintf('%d|%g', n_gp, density);
    if ~isempty(CACHE) && isfield(CACHE, 'key') && strcmp(CACHE.key, key)
        G = CACHE.G;
        G.sigma2 = sigma2_ladder(opts);
        return;
    end

    p = add_paths();
    rng(0);
    [X, y]   = load_dataset_csv_or_mat(fullfile(p.gpDir, 'data', 'pumadyn32nm.csv'));
    [Xs, ~]  = standardize_data(X, y);
    sel      = randperm(size(Xs, 1), n_gp);
    Xn       = Xs(sel, :);
    ell      = estimate_median_lengthscale(Xn);
    K        = rbf_kernel_matrix(Xn, ell);

    % Threshold to the requested density, symmetrically.  The pattern is fixed
    % once here and never depends on sigma2 -- the hypothesis of Thm 3.1.
    off  = K - diag(diag(K));
    thr  = quantile(off(off > 0), 1 - density);
    Ks   = K .* (K >= thr);
    Ks   = (Ks + Ks') / 2;

    % Thresholding destroys positive definiteness, so restore it ONCE with a
    % diagonal shift.  The result is a legitimate kernel-ridge system
    % A(sigma2) = Ks + sigma2*I with Ks PSD, exactly the GP_train form -- and,
    % crucially, the sparsity pattern is fixed for the whole sigma2 sweep
    % (a diagonal shift touches only entries already in the pattern).
    lam_min = min(real(eig(Ks)));
    if lam_min < 0
        Ks = Ks + (-lam_min) * eye(n_gp);
    end
    Ks = sparse(Ks);

    G = struct();
    G.n       = n_gp;
    G.Ks      = Ks;
    G.ell     = ell;
    G.density = nnz(Ks) / n_gp^2;
    G.lam_min = lam_min;                       % before the PSD shift
    G.sigma2  = sigma2_ladder(opts);
    CACHE = struct('key', key, 'G', G);
end

function s2 = sigma2_ladder(opts)
%SIGMA2_LADDER  The regularization sweep.  Ks is already PSD, so no shift.
    s2 = getdef(opts, 'sigma2', []);
    if isempty(s2), s2 = logspace(-4, 0, 8); end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
