function out = deflated_cond_two_level(W, Tfun, TinvFun, tau, n, opts)
%DEFLATED_COND_TWO_LEVEL  Condition number of the two-level deflated operator.
%
%   out = deflated_cond_two_level(W, Tfun, TinvFun, tau, n, opts) computes
%   the extreme eigenvalues and condition number of
%       H = P^{1/2} * Tsym * P^{1/2},
%       P = (I - Q Q') + tau * Q * (Q' Tsym Q)^{-1} * Q',
%   where Q is an orthonormal basis of range(W) and Tsym is the SPD operator
%   applied by Tfun.  This is the split-operator two-level deflation scheme
%   (same P^{1/2} form as src.precond.deflation_Psqrt_apply, applied on the
%   preconditioned operator Ahat = Tsym).
%
%   LOCAL trial version; once validated it may be promoted to +src/+precond.
%
%   With tau = lam_{k+1}(Tsym) and Q spanning the exact smallest-k
%   eigenvectors, spec(H) = {tau} U {lam_{k+1..n}}, so
%   kappa(H) = lam_max(Tsym) / lam_{k+1} -- the best achievable value.
%   Deflating with an approximate subspace can only increase kappa(H); the
%   ratio against the exact value quantifies the solver-side cost of the
%   capture error.
%
%   The extremes come from eigs on operator handles:
%     lam_max : 'largestabs' on H (SPD, well separated -- fast).
%     lam_min : 'largestabs' on H^{-1} = P^{-1/2} Tsym^{-1} P^{-1/2}, using
%               the EXACT inverse TinvFun; both P^{+-1/2} share one small
%               eigendecomposition of E = Q' Tsym Q (block-diagonal in the
%               Q basis: tau*E^{-1} on span(Q), identity on the complement).
%
%   Inputs
%     W       : n-by-r0 deflation block.  Need NOT be orthonormal or full
%               rank: orthonormalized + rank-truncated internally via
%               column-pivoted QR (same pattern as subspace_capture_directed).
%     Tfun    : @(X) Tsym * X   forward apply (SPD).
%     TinvFun : @(X) Tsym^{-1} * X   exact inverse apply.
%     tau     : positive scalar shift for the coarse correction.
%     n       : problem dimension.
%     opts    : optional struct:
%                 .W_is_orth  (default false) caller guarantees W already has
%                             orthonormal, full-rank columns; skips the QR.
%                 .eigs_tol   (default 1e-8)
%                 .eigs_maxit (default 5000)
%
%   Output (struct) -- numerical failures never throw, they set ok = false:
%     kappa    : lam_max / lam_min of H (NaN on failure).
%     lam_max  : largest eigenvalue of H (NaN on failure).
%     lam_min  : smallest eigenvalue of H (NaN on failure).
%     r        : numerical rank of W actually used for deflation.
%     ok       : true iff both extremes were computed.
%     err      : first line of the failure message ('' on success).

    if nargin < 6 || isempty(opts)
        opts = struct();
    end
    W_is_orth  = isfield(opts, 'W_is_orth')  && opts.W_is_orth;
    eigs_tol   = get_opt(opts, 'eigs_tol',   1e-8);
    eigs_maxit = get_opt(opts, 'eigs_maxit', 5000);

    if ~(isscalar(tau) && isnumeric(tau) && isfinite(tau) && tau > 0)
        error('deflated_cond_two_level:badTau', ...
              'tau must be a positive finite scalar.');
    end
    if ~isempty(W) && size(W, 1) ~= n
        error('deflated_cond_two_level:dimMismatch', ...
              'W has %d rows but n = %d.', size(W, 1), n);
    end

    out = struct('kappa', NaN, 'lam_max', NaN, 'lam_min', NaN, ...
                 'r', NaN, 'ok', false, 'err', '');

    % --- Orthonormal deflation basis (rank-truncating pivoted QR) ----------
    if W_is_orth
        Q = full(W);  r = size(W, 2);
    else
        [Q, r] = local_orth(full(W));
    end
    out.r = r;

    try
        % --- Coarse matrix E = Q' Tsym Q and its eigendecomposition --------
        % r == 0 degenerates cleanly: Q is n-by-0, the rank-r corrections
        % below vanish and H reduces to Tsym itself.
        TQ = Tfun(Q);
        E  = Q' * TQ;
        E  = full((E + E') / 2);
        [U, d] = eig(E, 'vector');
        d = real(d);
        if any(d <= 0)
            out.err = 'coarse matrix Q''*Tsym*Q is not numerically SPD';
            return;
        end

        % --- P^{+-1/2} applies (block-diagonal in the Q basis) -------------
        %   P^{ 1/2} = (I - QQ') + sqrt(tau)   * Q E^{-1/2} Q'
        %   P^{-1/2} = (I - QQ') + 1/sqrt(tau) * Q E^{ 1/2} Q'
        sq_d     = sqrt(d);
        PsqrtFun = @(X) X - Q * (Q' * X) ...
                      + sqrt(tau) * (Q * (U * ((1 ./ sq_d) .* (U' * (Q' * X)))));
        PinvSqrtFun = @(X) X - Q * (Q' * X) ...
                      + (1 / sqrt(tau)) * (Q * (U * (sq_d .* (U' * (Q' * X)))));

        % --- Extreme eigenvalues via eigs on handles ------------------------
        % Name-value options, NOT the legacy opts struct: eigs ignores
        % unknown struct fields, so a struct with these capitalized names
        % silently runs unsymmetric Arnoldi (issym=false), which stalls on
        % the degenerate top eigenvalue of H^{-1} (multiplicity = r).
        Hfun    = @(x) PsqrtFun(Tfun(PsqrtFun(x)));
        lam_max = real(eigs(Hfun, n, 1, 'largestabs', ...
                            'Tolerance', eigs_tol, ...
                            'MaxIterations', eigs_maxit, ...
                            'IsFunctionSymmetric', true));
        HinvFun = @(x) PinvSqrtFun(TinvFun(PinvSqrtFun(x)));
        mu      = real(eigs(HinvFun, n, 1, 'largestabs', ...
                            'Tolerance', eigs_tol, ...
                            'MaxIterations', eigs_maxit, ...
                            'IsFunctionSymmetric', true));
        if isempty(lam_max) || isempty(mu) || ~isfinite(lam_max) || ...
                ~isfinite(mu) || mu <= 0
            out.err = 'eigs did not converge on the deflated operator';
            return;
        end

        out.lam_max = lam_max;
        out.lam_min = 1 / mu;
        out.kappa   = lam_max * mu;
        out.ok      = true;
    catch ME
        out.err = regexprep(ME.message, '\n.*', '');
    end
end

function v = get_opt(opts, name, default)
%GET_OPT  Field of opts with a default.
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end

function [Q, r] = local_orth(V)
%LOCAL_ORTH  Orthonormal basis of range(V) with numerical rank truncation.
%   Column-pivoted economy QR: |diag(R)| is nonincreasing, so truncating at
%   the rank keeps the columns of Q that actually span range(V).
%   (Duplicated from subspace_capture_directed.m, where it is file-local.)

    if isempty(V)
        Q = zeros(size(V));
        r = 0;
        return;
    end

    [Q0, R, ~] = qr(V, 0);           % 3-output economy QR => column-pivoted

    d   = abs(diag(R));
    tol = max(size(V)) * eps(max(d));
    r   = sum(d > tol);
    Q   = Q0(:, 1:r);
end
