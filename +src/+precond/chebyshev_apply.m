function v = chebyshev_apply(A, r, m, lam_min, lam_max)
%CHEBYSHEV_APPLY  Apply T_m(Abar) to r via the three-term recurrence.
%
%   v = chebyshev_apply(A, r, m, lam_min, lam_max)
%
%   T_m is the degree-m Chebyshev polynomial of the first kind. Abar is the
%   affinely rescaled operator
%       Abar = (2*A - (lam_max + lam_min)*I) / (lam_max - lam_min),
%   so that the interval [lam_min, lam_max] maps to [-1, 1].
%
%   Eigenvalues of A inside [lam_min, lam_max] are damped by T_m (|T_m| <= 1
%   on [-1, 1]). Eigenvalues outside that interval are amplified
%   (|T_m(x)| ~ cosh(m*acosh|x|) for |x| > 1). Choosing [lam_min, lam_max]
%   to bracket the upper part of A's spectrum therefore turns T_m(Abar)
%   into a high-pass filter that selects the small-eigenvalue subspace --
%   the subspace one wants for a deflation/coarse prolongator.
%
%   Inputs
%     A         n-by-n numeric matrix OR function handle X -> A*X.
%               Handle must accept block input (n-by-k matrix).
%     r         n-by-1 vector or n-by-k block (residual / random start).
%     m         non-negative integer, polynomial degree. Costs m mat-vecs.
%     lam_min   scalar, lower end of the interval mapped to -1.
%     lam_max   scalar, upper end of the interval mapped to +1.
%               Requires lam_min < lam_max.
%
%   Output
%     v         T_m(Abar) * r, same shape as r. No internal normalization.
%
%   Implementation notes
%     - Three-term recurrence  T_{k+1}(x) = 2*x*T_k(x) - T_{k-1}(x), so the
%       loop carries only two n-by-k buffers regardless of m.
%     - When |lam_min| is well above the smallest eigenvalues of A, the norm
%       of the result grows like cosh(m * acosh(|x|)) and may overflow for
%       large m. Caller should pick a moderate m or normalize the output
%       (e.g. orth([V, v]) when appending to a basis V).

    if ~(isscalar(m) && m == floor(m) && m >= 0)
        error('chebyshev_apply:badDegree', ...
              'm must be a non-negative integer.');
    end
    if ~(isscalar(lam_min) && isscalar(lam_max) && ...
         isfinite(lam_min) && isfinite(lam_max) && lam_min < lam_max)
        error('chebyshev_apply:badInterval', ...
              'Need finite scalar lam_min < lam_max.');
    end

    if isnumeric(A)
        Afun = @(X) A * X;
    else
        Afun = A;
    end

    c = 0.5 * (lam_max + lam_min);
    d = 0.5 * (lam_max - lam_min);

    t_prev = r;                                  % T_0(Abar) * r = r
    if m == 0
        v = t_prev;
        return;
    end

    t_curr = (Afun(r) - c * r) / d;              % T_1(Abar) * r = Abar * r
    for k = 2:m
        t_next = (2 / d) * (Afun(t_curr) - c * t_curr) - t_prev;
        t_prev = t_curr;
        t_curr = t_next;
    end
    v = t_curr;
end
