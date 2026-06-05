function Hfun = hessian_operator(X, w, lambda)
%HESSIAN_OPERATOR  Matrix-free logistic-regression Newton Hessian H*v.
%   Hfun = HESSIAN_OPERATOR(X, w, lambda) returns a function handle that
%   applies the penalized logistic Hessian
%
%       H = Xa' * diag(w) * Xa + lambda * I ,   Xa = [X, ones(n,1)]
%
%   to a vector v of length d+1 = [weights; bias], WITHOUT forming H. Here
%   w = p.*(1-p) are the IRLS weights and the intercept is the implicit
%   trailing all-ones column. H is symmetric positive definite for lambda > 0,
%   so the handle is suitable for PCG and for EIGS (IsFunctionSymmetric).
%
%   This mirrors the operator-based eigs diagnostics in
%   GP_train/plot_kernel_spectrum.m and the TRON/LIBLINEAR Hessian-vector
%   product used to solve these high-dimensional systems.
%
%   See also logreg_newton, hessian_spectrum, pcg, eigs.

    Hfun = @(v) hess_apply(v, X, w, lambda);
end

%% --------- local helpers ---------
function Hv = hess_apply(v, X, w, lambda)
%HESS_APPLY  Apply H = [X 1]' diag(w) [X 1] + lambda*I to v = [vw; b].
    d  = numel(v) - 1;
    vw = v(1:d);
    b  = v(end);
    Xv = X * vw + b;            % n-by-1, includes intercept contribution
    t  = w .* Xv;               % n-by-1
    Hv = [X' * t + lambda * vw; sum(t) + lambda * b];
end
