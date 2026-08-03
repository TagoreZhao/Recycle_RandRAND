function [U, lam, info] = pencil_subspace(A, M, k, opts)
%PENCIL_SUBSPACE  The k smallest-|lambda| invariant subspace of the pencil (A,M).
%
%   [U, lam, info] = PENCIL_SUBSPACE(A, M, K)
%   [U, lam, info] = PENCIL_SUBSPACE(A, M, K, OPTS)
%
%   Solves A*u = lambda*M*u for the K eigenvalues of smallest modulus and
%   returns an orthonormal basis U of the corresponding invariant subspace --
%   the PHYSICAL deflation target, denoted U_k(A,M) in the document.  Only
%   span(U) is meaningful; the particular basis returned is an implementation
%   detail that no caller may depend on.
%
%   By Thm 1.1, V = C'*U is the matching invariant subspace of the split
%   operator Ahat = C^-1 A C^-T whenever M = C*C'.
%
%   OPTS:
%     .mode      'auto' (default) | 'dense' | 'eigs'
%                'auto' uses a dense generalized eig below .dense_max, which is
%                exact and immune to eigs' shift-invert trouble near lambda = 0
%                (the indefinite KKT pencil has eigenvalues on both sides of 0).
%     .dense_max dense threshold, default 2000
%     .tol       eigs Tolerance, default 1e-10
%     .maxit     eigs MaxIterations, default 1000
%
%   NOTE eigs is called with NAME-VALUE options.  Passing a struct (the older
%   syntax) is silently ignored by modern eigs, which then loses symmetric mode
%   -- a real hazard this repo has hit before.
%
%   info: .mode .n .k_requested .k_returned .lam_all(k+2 smallest) .gap
%         .gap = |lambda_{k+1}| - |lambda_k|, the Davis-Kahan denominator.
%
%   See also: gap, gap_M, src.precond.build_deflation_V.

    if nargin < 4 || isempty(opts), opts = struct(); end
    mode      = getdef(opts, 'mode',      'auto');
    dense_max = getdef(opts, 'dense_max', 2000);
    tol       = getdef(opts, 'tol',       1e-10);
    maxit     = getdef(opts, 'maxit',     1000);

    n = size(A, 1);
    A = (A + A') / 2;
    M = (M + M') / 2;

    if strcmp(mode, 'auto')
        mode = 'dense';
        if n > dense_max, mode = 'eigs'; end
    end

    switch mode
        case 'dense'
            [Vf, Df] = eig(full(A), full(M), 'chol');
            lam_all  = real(diag(Df));
            [~, ord] = sort(abs(lam_all), 'ascend');
            U        = Vf(:, ord(1:k));
            lam      = lam_all(ord(1:k));
            lam_ext  = lam_all(ord(1:min(k + 2, n)));

        case 'eigs'
            [Vf, Df] = eigs(A, M, k + 1, 'smallestabs', ...
                            'Tolerance', tol, 'MaxIterations', maxit, ...
                            'IsSymmetricDefinite', true);
            lam_ext  = real(diag(Df));
            [~, ord] = sort(abs(lam_ext), 'ascend');
            Vf       = Vf(:, ord);
            lam_ext  = lam_ext(ord);
            U        = Vf(:, 1:k);
            lam      = lam_ext(1:k);

        otherwise
            error('pencil_subspace:mode', 'unknown mode "%s"', mode);
    end

    U = orth_trunc(real(U));

    info = struct();
    info.mode        = mode;
    info.n           = n;
    info.k_requested = k;
    info.k_returned  = size(U, 2);
    info.lam         = lam(:);
    info.lam_ext     = lam_ext(:);
    if numel(lam_ext) > k
        info.gap = abs(lam_ext(k + 1)) - abs(lam_ext(k));
    else
        info.gap = NaN;
    end
end

function v = getdef(s, f, d)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
