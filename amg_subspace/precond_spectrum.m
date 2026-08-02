function out = precond_spectrum(Mfun, R, n, opts)
%PRECOND_SPECTRUM  Ritz spectrum of a preconditioned SPD operator.
%
%   out = precond_spectrum(Mfun, R, n, opts) estimates the spectrum of the
%   preconditioned operator M*A for an SPD system A = R'*R, where Mfun applies
%   the (symmetric, positive definite) preconditioner M.
%
%   Method.  M*A is not symmetric, but the similar operator
%       H = R * M * R'                 (H = R(MR') ~ (MR')R = M*A)
%   IS symmetric whenever M is, and is SPD whenever M is.  So the routine runs
%   symmetric Lanczos with full (twice) reorthogonalization on
%       Hfun = @(x) R*(Mfun(R'*x)),
%   accumulating the tridiagonal T on the fly (no extra operator applies), and
%   returns the Ritz values eig(T).
%
%   Why Lanczos and not eigs.  eigs('smallestabs') needs a handle applying
%   H^{-1}, i.e. M^{-1}; an AMG V-cycle has no inverse to apply, so the bottom
%   of the spectrum is unreachable that way.  Ritz values are also exactly the
%   spectral information CG itself converges against, and one code path covers
%   every preconditioner in the study -- deflation projectors, AMG V-cycles,
%   combined C_tau operators, ichol, and the unpreconditioned reference -- so
%   their numbers are directly comparable.
%
%   Accuracy caveat.  Ritz values converge to the TOP of the spectrum faster
%   than the bottom, so lam_min (and hence kappa) is the fragile number.  Use
%   out.m_used together with a second run at a smaller m to gauge drift, and
%   cross-check deflation arms against subspace_capture/deflated_cond_two_level,
%   which gets lam_min exactly when an exact A^{-1} is available.
%
%   Inputs
%     Mfun : @(X) M*X, symmetric positive definite preconditioner apply.
%            [] means M = I, giving the spectrum of A itself.  The handle need
%            not be block-capable; it is only ever called on single vectors.
%     R    : upper-triangular Cholesky factor of A, A = R'*R (sparse; applied
%            as matvecs R'*x and R*y, never densified).
%     n    : problem dimension.
%     opts : optional struct
%              .m       (default 300)  Lanczos steps; capped at n
%              .n_tail  (default 200)  how many extreme Ritz values to return
%              .v0      (default [])   starting vector; default is a fixed
%                                      rng(0) Gaussian so runs are reproducible
%                                      and every arm sees the same start
%              .reorth_passes (default 2)  full reorthogonalization passes
%              .breakdown_tol (default 1e-14) relative beta breakdown cutoff
%              .sing_tol      (default 1e-12) lam_min/lam_max below which M is
%                                      declared numerically singular and the
%                                      result is reported as a failure -- a
%                                      rank-deficient preconditioner (a pure
%                                      coarse-grid correction, say) otherwise
%                                      yields a meaningless kappa ~ 1/eps
%
%   Output (struct) -- numerical failures never throw, they set ok = false:
%     lam_min, lam_max : extreme Ritz values (NaN on failure)
%     kappa            : lam_max / lam_min   (NaN on failure)
%     ritz             : all Ritz values, ascending
%     ritz_low         : the smallest min(n_tail, numel(ritz)) Ritz values
%     ritz_high        : the largest  min(n_tail, numel(ritz)) Ritz values
%     m_used           : Lanczos steps actually taken (< m on lucky breakdown)
%     ok               : true iff a spectrum was produced
%     err              : first line of the failure message ('' on success)
%
%   See also DEFLATED_COND_TWO_LEVEL, RUN_AMG_DEFLATION_VS_PRECOND.

    if nargin < 4 || isempty(opts), opts = struct(); end
    m             = get_opt(opts, 'm',             300);
    n_tail        = get_opt(opts, 'n_tail',        200);
    v0            = get_opt(opts, 'v0',            []);
    reorth_passes = get_opt(opts, 'reorth_passes', 2);
    breakdown_tol = get_opt(opts, 'breakdown_tol', 1e-14);
    sing_tol      = get_opt(opts, 'sing_tol',      1e-12);

    validateattributes(n, {'numeric'}, {'scalar', 'integer', 'positive'});
    m = min(max(round(m), 1), n);

    out = struct('lam_min', NaN, 'lam_max', NaN, 'kappa', NaN, ...
                 'ritz', [], 'ritz_low', [], 'ritz_high', [], ...
                 'm_used', 0, 'ok', false, 'err', '');

    if isempty(Mfun)
        Mfun = @(x) x;                       % unpreconditioned reference
    elseif ~isa(Mfun, 'function_handle')
        error('precond_spectrum:badMfun', 'Mfun must be [] or a function handle.');
    end
    Rt   = R.';
    Hfun = @(x) R * (Mfun(Rt * x));

    try
        if isempty(v0)
            % Fixed start so every arm's Ritz values come from the same
            % Krylov seed -- differences are then the operator's, not the
            % starting vector's.
            rs = rng; cleanupRng = onCleanup(@() rng(rs));   %#ok<NASGU>
            rng(0);
            v0 = randn(n, 1);
        end
        nv0 = norm(v0);
        if ~(isfinite(nv0) && nv0 > 0)
            error('precond_spectrum:badV0', 'Starting vector is zero or non-finite.');
        end

        [alpha, beta, m_used] = lanczos_tridiag(Hfun, v0 / nv0, m, ...
                                                reorth_passes, breakdown_tol);
        out.m_used = m_used;
        if m_used < 1
            error('precond_spectrum:noSteps', 'Lanczos produced no usable steps.');
        end

        T    = diag(alpha(1:m_used));
        if m_used > 1
            T = T + diag(beta(1:m_used-1), 1) + diag(beta(1:m_used-1), -1);
        end
        ritz = sort(real(eig((T + T.') / 2)), 'ascend');

        out.ritz      = ritz;
        out.lam_min   = ritz(1);
        out.lam_max   = ritz(end);
        out.kappa     = out.lam_max / out.lam_min;
        nt            = min(n_tail, numel(ritz));
        out.ritz_low  = ritz(1:nt);
        out.ritz_high = ritz(end-nt+1:end);
        if ~(isfinite(out.kappa) && out.lam_min > 0)
            out.err = sprintf(['non-positive or non-finite extremes ', ...
                               '(lam_min=%g, lam_max=%g)'], ...
                              out.lam_min, out.lam_max);
        elseif out.lam_min <= sing_tol * out.lam_max
            % A rank-deficient M (e.g. a pure coarse-grid correction, whose
            % rank is coarseN) leaves lam_min at roundoff, so kappa comes out
            % ~1/eps: a number, but a meaningless one, and pcg on such an M is
            % not a valid method.  Report it as a failure with the reason
            % rather than publishing 1e16 as a condition number.
            out.err = sprintf(['preconditioner is numerically singular ', ...
                               '(lam_min/lam_max = %.3g)'], ...
                              out.lam_min / out.lam_max);
        else
            out.ok = true;
        end
    catch ME
        out.err = regexprep(ME.message, '\n.*', '');
    end
end

%% =========================================================================
%% Local helpers
%% =========================================================================
function [alpha, beta, j] = lanczos_tridiag(Hfun, q1, m, passes, btol)
%LANCZOS_TRIDIAG  Symmetric Lanczos with full reorthogonalization.
%   Returns the diagonal alpha(1:j) and off-diagonal beta(1:j-1) of the
%   tridiagonal projection, where j <= m is the number of steps completed
%   before (lucky) breakdown.  Q is kept in full for the reorthogonalization,
%   which is affordable at the m <= 300 used here and is what keeps the small
%   Ritz values trustworthy.
    n     = numel(q1);
    Q     = zeros(n, m);
    alpha = zeros(m, 1);
    beta  = zeros(max(m - 1, 1), 1);
    Q(:, 1) = q1;
    anorm   = 0;                              % running ||T|| for the beta test

    for j = 1:m
        w        = Hfun(Q(:, j));
        alpha(j) = Q(:, j)' * w;
        w        = w - alpha(j) * Q(:, j);
        if j > 1
            w = w - beta(j-1) * Q(:, j-1);
        end
        % Full reorthogonalization, repeated: one pass loses orthogonality
        % again for clustered spectra, which is precisely this study's case.
        for p = 1:passes
            w = w - Q(:, 1:j) * (Q(:, 1:j)' * w);
        end

        anorm = max(anorm, abs(alpha(j)) + 2 * beta(max(j-1, 1)));
        if j == m, break; end

        beta(j) = norm(w);
        if beta(j) <= btol * max(anorm, 1)
            % Lucky breakdown: the Krylov space is exhausted (or the starting
            % vector was deficient).  The Ritz values from the j steps taken
            % are exact eigenvalues of H; stop rather than restart.
            break;
        end
        Q(:, j+1) = w / beta(j);
    end
end

function val = get_opt(opts, name, default)
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        val = opts.(name);
    else
        val = default;
    end
end
