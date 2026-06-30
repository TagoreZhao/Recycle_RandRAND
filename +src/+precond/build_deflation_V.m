function V = build_deflation_V(A, P, opts, dA)
%BUILD_DEFLATION_V  Construct a dense orthonormal deflation basis V for the
% split (smoothed) operator Ahat = C^-1 A C^-T, by one of several operations.
% This is the indefinite-Stokes analog of the report's V-selection switch
% (Preconditioner_Recycle/report/solve_deflate_M_P*): add a V-building
% operation by adding a `case`.
%
%   V = build_deflation_V(A, P, opts, dA)
%
% The coarse space is the `sm_eig` smallest-|lambda| modes of Ahat (the ones
% that stall MINRES) plus, when `lg_eig > 0`, the `lg_eig` largest-|lambda|
% modes, concatenated and orthonormalized (mirrors the report's sm_eig/lg_eig).
%
% Inputs
%   A     n-by-n symmetric indefinite KKT matrix (the current-step system).
%   P     incomplete-LDL struct from make_ildl_precond(A,...), providing the
%         split-factor applies .applyCinv (C^-1), .applyCtinv (C^-T) and the
%         raw factors .L, .s, .p, .Dsqrt used to rebuild C = S^-1 P^T L Dsqrt.
%   opts  struct:
%           .method        'exact' | 'gaussian' | 'sjlt' | 'polynomial'
%           .sm_eig        # smallest-|lambda| deflation vectors  (>= 1)
%           .lg_eig        # largest-|lambda|  deflation vectors  (>= 0, default 0)
%           .q             sketch power-iteration steps           (default 2)
%           .cheb_degree   Chebyshev degree, 'polynomial'         (default 12)
%   dA    optional precomputed decomposition(A) reused by the inverse-power
%         sketch methods ('gaussian'/'sjlt'); [] -> build internally.
%
% Output
%   V     n-by-(sm_eig+lg_eig) real matrix with orthonormal columns (V'V = I).
%
% Methods (small basis)
%   'exact'       smallest-|lambda| eigvecs of Ahat via the generalized eig
%                 (A,M), M = C C':  Vs = C' * U.
%   'gaussian'    Gaussian sketch + plain power iteration on the EXACT inverse
%                 Ahat^-1 = C' A^-1 C  (one factorization dA = decomposition(A)).
%   'sjlt'        as 'gaussian' but with a sparse SJLT start block.
%   'polynomial'  Chebyshev high-pass on the SQUARED split operator Ahat^2
%                 (eigenvalues in [0, lam_max^2]); the near-zero-|lambda|
%                 indefinite cluster maps to the low end and is amplified.
%                 (A one-sided high-pass on the indefinite Ahat itself cannot
%                  isolate a mid-spectrum cluster — hence the square.)  The
%                 reject band is set from EXACT eigenvalues of (A,M) (the
%                 (sm_eig+1)-th smallest and the largest |lambda|), so only the
%                 filter is matrix-free; the bounds use eigs (report-style).
% Large basis (when lg_eig > 0): 'exact' uses eigs(A,M,'largestabs'); the
% sketched/polynomial methods use forward power iteration on Ahat (which
% converges to the largest-|lambda| modes).
%
% See also: make_ildl_precond, deflation_P_apply_indef, two_level_split_solve,
%           src.precond.subspace_iter_plain, src.precond.sjlt,
%           src.precond.chebyshev_apply.

    import src.precond.*

    if ~isfield(opts, 'method') || isempty(opts.method)
        error('build_deflation_V:noMethod', 'opts.method is required.');
    end
    if ~isfield(opts, 'sm_eig') || isempty(opts.sm_eig) || opts.sm_eig < 1
        error('build_deflation_V:badSmEig', 'opts.sm_eig (>=1) is required.');
    end
    n      = size(A, 1);
    method = opts.method;
    sm     = opts.sm_eig;
    lg     = getfield_default(opts, 'lg_eig', 0);
    q      = getfield_default(opts, 'q', 2);
    if lg < 0, error('build_deflation_V:badLgEig', 'opts.lg_eig must be >= 0.'); end

    % Explicit factor C (M = C C') is used by every method: the eig methods
    % ('exact', and the exact-band 'polynomial') form M = C C', and the
    % inverse-power sketches ('gaussian'/'sjlt') form Ahat^-1 = C' A^-1 C.
    Sinv = spdiags(1 ./ P.s, 0, n, n);
    Pt   = sparse(P.p, (1:n)', 1, n, n);          % permutation P^T
    C    = Sinv * Pt * P.L * P.Dsqrt;             % M = C C'
    M    = [];   % built lazily by the eig methods (shared by small + large)

    % ---- small basis: the sm smallest-|lambda| modes ----------------------
    switch method
        case 'exact'
            M = C * C';  M = (M + M') / 2;
            [Us, ~] = eigs(A, M, sm, 'smallestabs', ...
                           struct('tol', 1e-6, 'maxit', 1000));
            Vs = C' * Us;

        case {'gaussian', 'sjlt'}
            if nargin < 4 || isempty(dA)
                dA = decomposition(A);                % exact A^-1 factorization
            end
            AinvFun = @(Y) C' * (dA \ (C * Y));       % Ahat^-1 = C' A^-1 C
            Vs = subspace_iter_plain(AinvFun, start_block(method, n, sm), q);

        case 'polynomial'
            % Exact Chebyshev reject band from the (A,M) generalized spectrum
            % (report-style; (A,M) shares the spectrum of Ahat = C^-1 A C^-T):
            %   lower edge = |lambda_{sm+1}|  (just above the sm-mode cluster)
            %   upper edge = max|lambda(Ahat)|
            % on the SQUARED operator Ahat^2 (eigenvalues lambda^2 >= 0), so the
            % high-pass damps the bulk [lo,hi] and amplifies lambda^2 < lo, i.e.
            % the sm smallest-|lambda| deflation targets.
            M  = C * C';  M = (M + M') / 2;
            eo = struct('tol', 1e-6, 'maxit', 1000);
            dlo     = eigs(A, M, sm + 1, 'smallestabs', eo);   % sm+1 smallest |lambda|
            lam_cut = max(abs(real(dlo)));                     % (sm+1)-th smallest = cluster edge
            lam_max = abs(real(eigs(A, M, 1, 'largestabs', eo)));
            Ahat  = @(Y) P.applyCinv(A * P.applyCtinv(Y));
            Ahat2 = @(Y) Ahat(Ahat(Y));
            lo    = lam_cut^2;  hi = lam_max^2;                % exact reject band on Ahat^2
            deg   = getfield_default(opts, 'cheb_degree', 12);
            Vs = chebyshev_apply(Ahat2, randn(n, sm), deg, lo, hi);

        otherwise
            error('build_deflation_V:unknownMethod', ...
                  'unknown method ''%s''', method);
    end

    % ---- large basis: the lg largest-|lambda| modes (optional) ------------
    if lg > 0
        if strcmp(method, 'exact')
            [Ul, ~] = eigs(A, M, lg, 'largestabs', ...
                           struct('tol', 1e-6, 'maxit', 1000));
            Vl = C' * Ul;
        else
            % forward power iteration on the split operator -> largest |lambda|
            % modes (raise q to better resolve a tightly-clustered large end).
            Ahat = @(Y) P.applyCinv(A * P.applyCtinv(Y));
            Vl = subspace_iter_plain(Ahat, start_block(method, n, lg), q);
        end
        V = [Vs, Vl];
    else
        V = Vs;
    end

    [V, ~] = qr(real(V), 0);
end

%==========================================================================
%  Local helpers
%==========================================================================
function Om = start_block(method, n, k)
%START_BLOCK  Random start block: sparse SJLT for 'sjlt', Gaussian otherwise.
    if strcmp(method, 'sjlt')
        Om = src.precond.sjlt(n, k, 8);
    else
        Om = randn(n, k);
    end
end

function v = getfield_default(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
