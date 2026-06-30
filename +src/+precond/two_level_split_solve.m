function [x, fl, rr, it] = two_level_split_solve(K, b, tol, mit, P, V, tau)
%TWO_LEVEL_SPLIT_SOLVE  Solve K x = b by MINRES on the split (smoothed) operator
% Ahat = C^-1 K C^-T with an optional indefinite deflation projector as the inner
% preconditioner, then recover x = C^-T y.  This is the standard two-level
% deflation scheme B = L^-T P L^-1 (L = C, the incomplete-LDL factor), the
% indefinite port of the report's solve_deflate_M_P.
%
%   [x, fl, rr, it] = two_level_split_solve(K, b, tol, mit, P, V, tau)
%
% Inputs
%   K        n-by-n symmetric indefinite KKT matrix (current step).
%   b        right-hand side.
%   tol,mit  MINRES tolerance and max iterations.
%   P        incomplete-LDL struct from make_ildl_precond (uses .applyCinv = C^-1,
%            .applyCtinv = C^-T).
%   V        dense orthonormal coarse basis for the split operator, or [] for the
%            ILDL-only solve (no inner preconditioner).
%   tau      deflation coarse-correction weight (multiplicative), e.g. 1.
%
% Outputs match MINRES: solution x, flag fl, relative residual rr (of the SPLIT
% operator), iteration count it.
%
% See also: build_deflation_V, deflation_P_apply_indef, make_ildl_precond.

    import src.precond.*

    Afun = @(y) P.applyCinv(K * P.applyCtinv(y));   % Ahat = C^-1 K C^-T
    btil = P.applyCinv(b);                          % C^-1 b

    if isempty(V)
        [y, fl, rr, it] = minres(Afun, btil, tol, mit);                 % ILDL only
    else
        Pdef = deflation_P_apply_indef(V, Afun, tau, 'handle', 0);      % (I-VV')+tau V|E|^-1 V'
        [y, fl, rr, it] = minres(Afun, btil, tol, mit, Pdef);          % two-level L^-T P L^-1
    end

    x = P.applyCtinv(y);                            % recover x = C^-T y
end
