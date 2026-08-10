function [x, d] = woodbury_naive(A, U, C, V, b, xexact)
%WOODBURY_NAIVE  The textbook Woodbury evaluation, written to be broken.
%   [X, D] = WOODBURY_NAIVE(A, U, C, V, B)
%   [X, D] = WOODBURY_NAIVE(A, U, C, V, B, XEXACT)
%
%   Solves (A + U C V) x = b by the Sherman-Morrison-Woodbury identity
%
%       x = A^{-1}b  -  A^{-1}U ( C^{-1} + V A^{-1} U )^{-1} V A^{-1} b
%
%   exactly as written, in the order written, with NO defensive measures: no
%   symmetrization, no equilibration, no orthogonalization of U or V, no iterative
%   refinement, no special-casing of small updates, and C^{-1} formed explicitly by
%   inv().  This is deliberate.  The identity is exact in exact arithmetic, so any
%   error this routine makes is an error of EVALUATION, and the point of the
%   diagnostics below is to say which of the two evaluation steps lost the digits.
%
%   THE TWO PLACES DIGITS GO MISSING.  Both are cancellation, and neither is
%   visible in a condition number of A or of A + UCV:
%
%     (1) the small matrix.  S = C^{-1} + V A^{-1} U is a SUM.  If its two terms
%         nearly cancel, S is formed with an absolute error ~eps*max(|term|) that
%         is a large RELATIVE error.  Measured by
%
%             cancel_S = (||C^{-1}|| + ||V|| ||A^{-1}U||) / ||S||
%
%     (2) the final subtraction.  x = z - Y w with z = A^{-1}b.  If ||z|| >> ||x||
%         the answer is the difference of two large nearly equal vectors and the
%         leading digits annihilate.  Measured by
%
%             cancel_sub = (||z|| + ||Y w||) / ||z - Y w||
%
%   Both factors are 1 when nothing cancels and grow without bound when it does.
%   The forward error is bounded by roughly cancel_sub * eps (see
%   tests/test_stress_metrics.m T9), which is what makes them predictive rather
%   than merely descriptive.
%
%   THIS IS A DENSE ROUTINE for small, explicitly formed systems: it calls cond()
%   and inv() and forms A + U*C*V.  It exists to characterize the identity, not to
%   solve anything at scale.  The sparse production path is woodbury_solve, which
%   evaluates the same expression in the same order against a frozen factorization.
%
%   Output D:
%     .kappa_A .kappa_M .kappa_S   cond of A, of A + U C V, and of the small matrix
%     .cancel_S .cancel_sub        the two cancellation factors above
%     .resid .relres               ||b - (A+UCV)x|| and its relative form
%     .bwd                         ||r|| / (||A+UCV|| ||x|| + ||b||), normwise
%     .fwd                         ||x - xexact||/||xexact||, NaN if not supplied
%     .S .w .z .Y                  the intermediates, for callers that want them
%
%   See also: woodbury_solve, dd_woodbury_scalar, run_woodbury_scalar_stress.

    if nargin < 6
        xexact = [];
    end

    % --- the identity, naively ------------------------------------------
    z    = A \ b;
    Y    = A \ U;
    Cinv = inv(C);                       % explicit, on purpose -- see the header
    S    = Cinv + V * Y;
    w    = S \ (V * z);
    Yw   = Y * w;
    x    = z - Yw;

    % --- diagnostics -----------------------------------------------------
    M = A + U * C * V;
    r = b - M * x;

    d            = struct();
    d.kappa_A    = cond(A);
    d.kappa_M    = cond(M);
    d.kappa_S    = cond(S);
    d.cancel_S   = local_ratio(norm(Cinv) + norm(V) * norm(Y), norm(S));
    d.cancel_sub = local_ratio(norm(z) + norm(Yw), norm(x));
    d.resid      = norm(r);
    d.relres     = d.resid / max(norm(b), realmin);
    d.bwd        = d.resid / (norm(M) * norm(x) + norm(b));
    if isempty(xexact)
        d.fwd = NaN;
    else
        d.fwd = norm(x - xexact) / max(norm(xexact), realmin);
    end
    d.S = S;
    d.w = w;
    d.z = z;
    d.Y = Y;
end

%==========================================================================
function r = local_ratio(num, den)
%LOCAL_RATIO  num/den, with total cancellation reported as Inf rather than NaN.
%   den == 0 is the interesting case, not a degenerate one: it means the
%   subtraction returned exactly zero and every digit was lost.
    if den > 0
        r = num / den;
    elseif num > 0
        r = Inf;
    else
        r = 1;
    end
end
