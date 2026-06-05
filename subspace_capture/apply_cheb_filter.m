function Y = apply_cheb_filter(A, X, filt)
%APPLY_CHEB_FILTER Apply a Chebyshev polynomial filter to a vector or block.
%
%   Y = apply_cheb_filter(A, X, filt)
%
% Inputs
%   A     matrix or function handle.
%         If matrix:          A * X is used.
%         If function handle: A(X) must return A*X.
%
%   X     n-by-s input block.
%
%   filt  struct returned by make_balanced_cheb_filter.
%
% Output
%   Y     rho_k(Ahat) X, where Ahat = (A - c I)/d.
%
% The filter is:
%   rho_k(t) = sum_{ell=0}^k coeff(ell+1) T_ell(t).
%
% The recurrence is:
%   V_0 = X
%   V_1 = Ahat X
%   V_l = 2 Ahat V_{l-1} - V_{l-2}

    coeff = filt.coeff(:);
    k = filt.k;
    c = filt.c;
    d = filt.d;

    if d <= 0
        error('Invalid filter scaling: filt.d must be positive.');
    end

    % Matrix-vector or operator application.
    if isa(A, 'function_handle')
        applyA = @(V) A(V);
    else
        applyA = @(V) A * V;
    end

    applyAhat = @(V) (applyA(V) - c * V) / d;

    % Degree 0.
    V0 = X;
    Y = coeff(1) * V0;

    if k == 0
        return;
    end

    % Degree 1.
    V1 = applyAhat(V0);
    Y = Y + coeff(2) * V1;

    % Degrees 2,...,k.
    for ell = 2:k
        V2 = 2 * applyAhat(V1) - V0;
        Y = Y + coeff(ell+1) * V2;

        V0 = V1;
        V1 = V2;
    end
end