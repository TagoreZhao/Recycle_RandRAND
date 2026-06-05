function Y = min_subspace_iter(A, P, q, Dinv, omega, reorth)
%MIN_SUBSPACE_ITER  Damped-Jacobi subspace iteration for the smallest M-eigvecs.
%
%   Y = min_subspace_iter(A, P, q, Dinv)
%   Y = min_subspace_iter(A, P, q, Dinv, omega)
%   Y = min_subspace_iter(A, P, q, Dinv, omega, reorth)
%
%   Iterates the damped-Jacobi power operator
%       T(Y) = ( I - omega * D^{-1} * A ) Y
%   on a starting block P for q outer steps. By default no
%   orthogonalization is performed inside the loop (plain power iteration);
%   a single orth(Y) is always applied before returning so the output is an
%   orthonormal basis. Set reorth=true to re-orthogonalize after every step.
%
%   A may be a numeric matrix or a function handle X -> A*X. When A is a
%   preconditioned operator (e.g. M = L^{-1} A_raw L^{-T}), pass the handle
%   directly and supply Dinv = 1 ./ diag(M). The function makes no
%   assumption about how A was preconditioned.
%
%   Inputs
%     A       n-by-n SPD matrix OR function handle X -> A*X (handles block X).
%     P   n-by-m starting block (random or oblivious sketch).
%     q       non-negative integer; number of outer power-iteration steps.
%     Dinv    n-by-1 vector of diagonal Jacobi weights (e.g. 1 ./ diag(A)).
%     omega   scalar damping coefficient. Default 2/3.
%     reorth  logical; if true, orth(Y) after every step. Default false.
%
%   Output
%     Y       n-by-m orthonormal basis after q steps and one final orth.
%
%   Example
%     M_apply = @(X) L \ (A_raw * (L' \ X));
%     Dinv    = 1 ./ diag_Ahat;            % diag of M, precomputed
%     Y       = min_subspace_iter(M_apply, randn(n, 2*k), 20, Dinv);

    if nargin < 5 || isempty(omega),  omega  = 2/3;   end
    if nargin < 6 || isempty(reorth), reorth = false; end

    if ~(isscalar(q) && q == floor(q) && q >= 0)
        error('min_subspace_iter:badQ', 'q must be a non-negative integer.');
    end

    if isnumeric(A)
        Afun = @(X) A * X;
    else
        Afun = A;
    end

    Y = P;
    for s = 1:q
        Y = Y - omega * (Dinv .* Afun(Y));
        if reorth
            Y = orth(Y);
        end
    end
    Y = orth(Y);
end
