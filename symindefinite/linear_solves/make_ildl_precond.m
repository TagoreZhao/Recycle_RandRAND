function P = make_ildl_precond(A, opts)
%MAKE_ILDL_PRECOND  SPD preconditioner from an incomplete LDL^T factorization
% of a symmetric indefinite matrix, for use with MINRES.
%
%   P = MAKE_ILDL_PRECOND(A)
%   P = MAKE_ILDL_PRECOND(A, opts)
%
%   MINRES requires a SYMMETRIC POSITIVE DEFINITE preconditioner.  An LDL^T
%   factorization of an indefinite A has an indefinite block-diagonal D, so we
%   form the SPD operator  M = S^-1 P^T L |D| L^T P S^-1  by taking the absolute
%   value of the 1x1 / 2x2 eigenvalues of D (the standard SYM-ILDL/MINRES trick),
%   where [L,D,p,S] = ldl(A) gives the fill-reducing permutation p and scaling S.
%   With C = S^-1 P^T L |D|^{1/2} we have M = C C^T, and the returned handles let
%   you either (a) apply M^-1 directly (MINRES 5th argument) or (b) form the
%   explicitly preconditioned/split system  C^-1 A C^-T.
%
%   "Incomplete" is the cheapest level-0 (no-fill) variant by default: the exact
%   factor L is restricted to the sparsity pattern of A (the LDL^T analog of
%   ichol('nofill')).  D is kept exact (it is cheap, block diagonal).
%
%   Inputs:
%     A    - n x n sparse symmetric indefinite matrix
%     opts - (optional) struct:
%              .mode    'nofill' (default) restrict L to pattern of A;
%                       'droptol' drop |L_ij| < droptol.
%              .droptol absolute drop tolerance for mode 'droptol' (default 1e-3).
%
%   Output struct P:
%     .applyMinv   @(r) -> M^-1 r     SPD preconditioner apply (MINRES 5th arg)
%     .applyCinv   @(r) -> C^-1 r     left factor solve  (split/transformed system)
%     .applyCtinv  @(y) -> C^-T y     right factor solve (recover x = C^-T y)
%     .L .Dsqrt .Disqrt .p .s         factors (Dsqrt=|D|^{1/2}, Disqrt=|D|^{-1/2})
%     .nnzL .fill_ratio               diagnostics (vs nnz(tril(A)))
%     .mode .droptol
%
%   See also: ldl, minres, test_ildl_minres, make_ildl_precond>abs_block_diag.

    if nargin < 2 || isempty(opts), opts = struct(); end
    if ~isfield(opts, 'mode')    || isempty(opts.mode),    opts.mode    = 'nofill'; end
    if ~isfield(opts, 'droptol') || isempty(opts.droptol), opts.droptol = 1e-3;    end

    n = size(A, 1);
    A = (A + A')/2;                       % enforce exact symmetry before factoring

    % --- exact LDL^T with fill-reducing permutation p and scaling S ----------
    [L, D, p, S] = ldl(sparse(A), 'vector');
    s = full(diag(S));                    % S is diagonal

    % --- incomplete restriction of L -----------------------------------------
    switch lower(opts.mode)
        case 'nofill'
            Aperm = S * A * S;            % the matrix actually factored is Aperm(p,p)
            Aperm = Aperm(p, p);
            % level-0 pattern; force the unit diagonal (A has zero-diagonal rows
            % in the constraint block, which would otherwise null L's diagonal).
            mask  = spones(tril(Aperm, -1)) + speye(n);
            L     = L .* mask;
        case 'droptol'
            keep = abs(L) >= opts.droptol;
            keep = keep | (speye(n) > 0); % always keep the unit diagonal
            L    = L .* keep;
        otherwise
            error('make_ildl_precond:mode', 'unknown opts.mode "%s"', opts.mode);
    end

    % --- SPD-ify D:  |D|^{1/2} and |D|^{-1/2} (1x1 / 2x2 blocks) --------------
    [Dsqrt, Disqrt] = abs_block_diag(D);

    % --- apply handles (M = C C^T, C = S^-1 P^T L |D|^{1/2}) ------------------
    % C^-1 r  = |D|^{-1/2} ( L \ (S r)(p) )
    applyCinv = @(r) Disqrt * ( L \ scatter_fwd(s .* r, p) );
    % C^-T y  = S * scatter_back( L' \ (|D|^{-1/2} y), p )
    applyCtinv = @(y) s .* scatter_back(L' \ (Disqrt * y), p, n);
    applyMinv  = @(r) applyCtinv(applyCinv(r));

    P = struct();
    P.applyMinv  = applyMinv;
    P.applyCinv  = applyCinv;
    P.applyCtinv = applyCtinv;
    P.L = L;  P.Dsqrt = Dsqrt;  P.Disqrt = Disqrt;  P.p = p;  P.s = s;
    P.nnzL = nnz(L);
    P.fill_ratio = nnz(L) / max(nnz(tril(A)), 1);
    P.mode = lower(opts.mode);  P.droptol = opts.droptol;
end

%==========================================================================
%  Helpers
%==========================================================================
function y = scatter_fwd(x, p)
%SCATTER_FWD  Forward permutation y = P x = x(p).
    y = x(p);
end

function y = scatter_back(x, p, n)
%SCATTER_BACK  Inverse permutation y = P^T x  (y(p) = x).
    y = zeros(n, 1);
    y(p) = x;
end

function [Dsqrt, Disqrt] = abs_block_diag(D)
%ABS_BLOCK_DIAG  |D|^{1/2} and |D|^{-1/2} for the block-diagonal D returned by
% ldl (1x1 and symmetric 2x2 blocks), via per-block eigenvalue absolute value.
    n = size(D, 1);
    floorv = 1e-14;                       % guard against exactly-singular blocks
    I = zeros(n + nnz(triu(D,1))*4, 1);   % generous preallocation
    J = I;  Vh = I;  Vi = I;  c = 0;

    k = 1;
    while k <= n
        is2x2 = (k < n) && (D(k+1, k) ~= 0);
        if is2x2
            B = full(D(k:k+1, k:k+1));
            B = (B + B')/2;
            [Q, Lam] = eig(B);
            a   = max(abs(diag(Lam)), floorv);
            Bh  = Q * diag(sqrt(a))  * Q';   % |B|^{1/2}
            Bi  = Q * diag(1./sqrt(a)) * Q'; % |B|^{-1/2}
            for ii = 0:1
                for jj = 0:1
                    c = c + 1;
                    I(c) = k+ii;  J(c) = k+jj;
                    Vh(c) = Bh(ii+1, jj+1);
                    Vi(c) = Bi(ii+1, jj+1);
                end
            end
            k = k + 2;
        else
            a   = max(abs(D(k, k)), floorv);
            c = c + 1;
            I(c) = k;  J(c) = k;
            Vh(c) = sqrt(a);  Vi(c) = 1/sqrt(a);
            k = k + 1;
        end
    end
    Dsqrt  = sparse(I(1:c), J(1:c), Vh(1:c), n, n);
    Disqrt = sparse(I(1:c), J(1:c), Vi(1:c), n, n);
end
