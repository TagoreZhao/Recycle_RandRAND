function [Papply, E, decE] = deflation_P_apply_indef(V, A, tau, output_type, RAND_EIGS)
%DEFLATION_P_APPLY_INDEF  Two-level deflation preconditioner for symmetric
% INDEFINITE systems (revisable copy of +src/+precond/deflation_P_apply).
%
%   [Papply, E, decE] = DEFLATION_P_APPLY_INDEF(V, A, tau, output_type, RAND_EIGS)
%
%   Builds the two-level deflation / coarse-correction operator
%
%       P = (I - V V') + tau * V |E|^{-1} V',     E = V' A V,
%
%   applied as  PX = X - V(V'X) + tau * V (|E|^{-1} (V'X)).
%
%   DIFFERENCE FROM THE ORIGINAL: the original +src/+precond/deflation_P_apply
%   factors the coarse matrix E with chol, which REQUIRES E (hence A on span(V))
%   to be SPD.  For a symmetric INDEFINITE A (e.g. a Stokes/KKT saddle-point
%   system) E = V'AV is indefinite and chol fails.  Here we instead use the SPD
%   inverse |E|^{-1}: eigendecompose E = W L W' and replace L by |L|, so
%   |E|^{-1} = W |L|^{-1} W' is SPD.  This is the same |.|-via-eigenvalue trick
%   used by make_ildl_precond>abs_block_diag for the |D| of an indefinite LDL^T.
%
%   Because |E|^{-1} is SPD and V is orthonormal, the returned operator P is
%   SPD (P = I on range(V)^perp and P = tau|E|^{-1} on range(V), both SPD).
%   It is therefore a LEGAL MINRES preconditioner, unlike the chol-based
%   original which is only valid in the SPD/PCG path.  With exact eigenvectors
%   E = diag(lambda_i) and the deflated eigenvalues are mapped to +/- tau,
%   clearing the small-|lambda| cluster that stalls MINRES.
%
%   Inputs:
%     V           - n x k deflation basis, assumed orthonormal (V'V ~ I).
%     A           - n x n symmetric matrix OR a function handle Afun(X) = A*X.
%     tau         - positive scalar weight on the coarse correction (default 1).
%     output_type - 'handle' (default) returns Papply as a function handle;
%                   'matrix' returns the dense n x n operator.
%     RAND_EIGS   - kept for signature parity with the original; the sketched
%                   struct path (RAND_EIGS==2) is NOT supported here (dense V).
%
%   Outputs:
%     Papply - function handle @(X) -> PX (or dense matrix if output_type
%              is 'matrix').
%     E      - the symmetrized coarse matrix V'AV (indefinite in general).
%     decE   - struct with fields E, W, absd (=|eig(E)|, floored), absEinv
%              (@(y)->|E|^{-1} y) and Qabs (@(X)->V|E|^{-1}V' X).
%
%   See also: deflation_P_apply (original SPD version), make_ildl_precond,
%   test_deflation_minres.

    if nargin < 3 || isempty(tau), tau = 1; end
    if tau <= 0, error('tau must be positive.'); end
    if ~(isnumeric(A) || isa(A,'function_handle'))
        error('A must be a numeric matrix or a function handle Afun(X)=A*X.');
    end
    if nargin < 4 || isempty(output_type), output_type = 'handle'; end
    if ~ismember(output_type, {'handle', 'matrix'})
        error('output_type must be ''handle'' or ''matrix''.');
    end
    if nargin < 5 || isempty(RAND_EIGS), RAND_EIGS = 0; end
    if RAND_EIGS == 2
        error(['deflation_P_apply_indef:sketchUnsupported', newline, ...
               'The sketched struct path (RAND_EIGS==2) is not supported in ' ...
               'the indefinite copy; pass a dense orthonormal matrix V.']);
    end

    % Build coarse matrix E = V'AV  (symmetric indefinite for indefinite A)
    AV = apply_A(A, V);
    E  = V' * AV;
    E  = (E + E')/2;                          % symmetrize before eig

    % SPD coarse inverse |E|^{-1} = W |L|^{-1} W'  (replaces chol / Z = V/R).
    [W, Lam] = eig(full(E));
    absd = abs(diag(Lam));
    absd = max(absd, 1e-14 * max(absd, [], 'all'));   % floor near-zero eigs
    if any(~isfinite(absd)) || max(absd) == 0
        error('Coarse matrix V''AV is numerically singular; |E|^{-1} undefined.');
    end

    absEinv = @(y) W * ((W' * y) ./ absd);    % |E|^{-1} applied
    Qabs    = @(X) V * absEinv(V' * X);       % V |E|^{-1} V'   (SPD)

    decE = struct();
    decE.E       = E;
    decE.W       = W;
    decE.absd    = absd;
    decE.absEinv = absEinv;
    decE.Qabs    = Qabs;

    % SPD two-level operator P = (I - VV') + tau * V |E|^{-1} V'
    if strcmp(output_type, 'handle')
        Papply = @(X) X - V*(V'*X) + tau * Qabs(X);
    else
        n = size(V, 1);
        Papply = eye(n) - V*V' + tau * (V * (W * diag(1./absd) * W') * V');
    end
end

function AX = apply_A(A, X)
    if isnumeric(A)
        AX = A * X;
        return;
    end
    try
        AX = A(X);
    catch
        [n,m] = size(X);
        AX = zeros(n,m, class(X));
        for j = 1:m
            AX(:,j) = A(X(:,j));
        end
    end
end
