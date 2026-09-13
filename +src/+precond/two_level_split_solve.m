function [x, fl, rr, it, info] = two_level_split_solve(K, b, tol, mit, P, V, tau)
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
% operator), iteration count it. Optional INFO contains timings, counted
% operator columns, coarse eigenvalues, AV and MATLAB's residual history.
% Four-output callers retain the original numerical path.
%
% See also: build_deflation_V, deflation_Psqrt_apply, make_ildl_precond.

    import src.precond.*

    measured = nargout > 4;
    operator_columns = 0;
    AV = [];
    d = [];
    Afun = @apply_split;
    setup_timer = tic;
    btil = P.applyCinv(b);                          % C^-1 b

    Pdef = [];
    if ~isempty(V)
        if measured
            [Pdef, ~, decE] = deflation_Psqrt_apply(V, @apply_squared, tau, 'handle');
            d = decE.d;
        else
            Pdef = deflation_Psqrt_apply(V, @(z) Afun(Afun(z)), tau, 'handle');
        end
    end
    setup_s = toc(setup_timer);
    setup_columns = operator_columns;
    solve_timer = tic;
    [y, fl, rr, it, rv] = minres(Afun, btil, tol, mit, Pdef);
    x = P.applyCtinv(y);                            % recover x = C^-T y
    solve_s = toc(solve_timer);
    if measured
        info = struct('coarse_setup_s', setup_s, 'minres_s', solve_s, ...
            'coarse_operator_columns', setup_columns, ...
            'minres_operator_columns', operator_columns - setup_columns, ...
            'coarse_eigenvalues', d, 'AV', AV, 'resvec', rv);
    end

    function z = apply_split(y)
        if measured, operator_columns = operator_columns + size(y,2); end
        z = P.applyCinv(K * P.applyCtinv(y));
    end
    function z = apply_squared(y)
        AV = Afun(y);
        z = Afun(AV);
    end
end
