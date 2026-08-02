function [x, fl, rr, it] = two_level_split_solve(K, b, tol, mit, P, V, tau)
%TWO_LEVEL_SPLIT_SOLVE  Solve K x = b by MINRES on the split (smoothed) operator
% Ahat = C^-1 K C^-T with an optional deflation coarse operator as the inner
% preconditioner, then recover x = C^-T y.  This is the standard two-level
% deflation scheme B = L^-T P L^-1 (L = C, the incomplete-LDL factor), the
% indefinite port of the report's solve_deflate_M_P.
%
% The coarse operator is the SQUARE ROOT of the SPD deflation preconditioner for
% the SQUARED split operator Ahat^2 (which is SPD, so E2 = V'Ahat^2 V > 0 needs
% no |.|-of-eigenvalues trick):
%   Pdef = (I - VV') + sqrt(tau) * V (V'Ahat^2 V)^{-1/2} V'   (SPD)
% built via src.precond.deflation_Psqrt_apply on Ahat2 = @(z) Afun(Afun(z)).  On
% span(V) it approaches (Ahat^2)^{-1/2} = |Ahat|^{-1}, the ideal SPD coarse
% correction for MINRES on the indefinite split operator.
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
% See also: build_deflation_V, deflation_Psqrt_apply, make_ildl_precond.

    import src.precond.*

    Afun = @(y) P.applyCinv(K * P.applyCtinv(y));   % Ahat = C^-1 K C^-T
    btil = P.applyCinv(b);                          % C^-1 b

    if isempty(V)
        [y, fl, rr, it] = minres(Afun, btil, tol, mit);                 % ILDL only
    else
        Ahat2 = @(z) Afun(Afun(z));                                     % Ahat^2 (SPD)
        Pdef  = deflation_Psqrt_apply(V, Ahat2, tau, 'handle');         % (I-VV')+sqrt(tau) V (V'Ahat^2V)^-1/2 V' ~ |Ahat|^-1
        %Compute 5 power iteration of Ahat2 and divided 2 
        [y, fl, rr, it] = minres(Afun, btil, tol, mit, Pdef);          % two-level L^-T P L^-1
    end

    x = P.applyCtinv(y);                            % recover x = C^-T y
end
