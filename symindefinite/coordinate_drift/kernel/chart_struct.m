function P = chart_struct(C)
%CHART_STRUCT  Wrap an explicit factor C as the chart interface the solvers use.
%
%   P = CHART_STRUCT(C)
%
%   src.precond.two_level_split_solve consumes only P.applyCinv and
%   P.applyCtinv, so any factor -- including one deliberately regauged to C*Q --
%   can be driven through the production solver by wrapping it here.  That is
%   what makes the gauge-covariance experiment (Thm 1.4) a test of the real
%   code path rather than of a reimplementation.
%
%   Dense solves by design: the regauged factor C*Q is dense, and these
%   experiments run at n <~ 2000.
%
%   See also: make_case, src.precond.two_level_split_solve.

    C = full(C);
    P = struct();
    P.C          = C;
    P.applyCinv  = @(r) C  \ r;
    P.applyCtinv = @(y) C' \ y;
    P.applyMinv  = @(r) C' \ (C \ r);
end
