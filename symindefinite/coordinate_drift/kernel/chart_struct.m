function P = chart_struct(C, defl_kind, tau)
%CHART_STRUCT  Wrap an explicit factor C as the chart interface the solvers use.
%
%   P = CHART_STRUCT(C)
%   P = CHART_STRUCT(C, DEFL_KIND, TAU)
%
%   src.precond.two_level_split_solve and two_level_solve_local consume only
%   P.applyCinv and P.applyCtinv, so any factor -- including one deliberately
%   regauged to C*Q -- can be driven through the production solver by wrapping it
%   here.  That is what makes the gauge-covariance experiment (Thm 1.4) a test of
%   the real code path rather than of a reimplementation.
%
%   DEFL_KIND ('spd' | 'indef') and TAU travel with the chart so that a regauged
%   copy of a case keeps its family's coarse-correction form; carry them over
%   from the case being regauged (cs.defl_kind, cs.tau).  Left empty, the
%   .defl_kind field is absent and two_level_solve_local will insist on an
%   explicit kind rather than guess.
%
%   Dense solves by design: the regauged factor C*Q is dense, and these
%   experiments run at n <~ 2000.
%
%   See also: make_case, two_level_solve_local, coarse_correction.

    C = full(C);
    P = struct();
    P.C          = C;
    P.applyCinv  = @(r) C  \ r;
    P.applyCtinv = @(y) C' \ y;
    P.applyMinv  = @(r) C' \ (C \ r);

    if nargin >= 2 && ~isempty(defl_kind), P.defl_kind = lower(defl_kind); end
    if nargin >= 3 && ~isempty(tau),       P.tau       = tau;              end
end
