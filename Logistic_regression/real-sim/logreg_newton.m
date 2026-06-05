function [beta, info, w] = logreg_newton(X, y, lambda, opts)
%LOGREG_NEWTON  Matrix-free Newton-CG (TRON-style) logistic regression.
%   [beta, info, w] = LOGREG_NEWTON(X, y, lambda, opts) minimizes the
%   L2-penalized logistic loss
%
%       f(b) = sum_i softplus(eta_i) - y_i*eta_i + 0.5*lambda*||b||^2 ,
%       eta = Xa*b ,  Xa = [X, ones(n,1)] ,  b = [weights; bias]
%
%   using Newton's method. Each Newton step solves the system
%   H*delta = -g with H = Xa'diag(p(1-p))Xa + lambda*I via PCG against the
%   matrix-free HESSIAN_OPERATOR (never forming H), followed by an Armijo
%   backtracking line search. The intercept is the implicit trailing all-ones
%   column and is regularized like the weights, so H >= lambda*I exactly
%   (the spectrum has a clean floor at lambda).
%
%   Inputs
%     X      : n-by-d (sparse) standardized features.
%     y      : n-by-1 in {0,1}.
%     lambda : L2 penalty (> 0).
%     opts   : struct, optional fields:
%                MaxIter (100), Tol (1e-8, relative grad-norm),
%                CGtol (1e-6), CGmaxit (500), Verbose (false).
%   Outputs
%     beta   : (d+1)-by-1 solution [weights; bias].
%     info   : struct (newton_iters, cg_iters_total, converged, train_acc,
%              separable, obj_trace, gnorm_trace, cg_iters_per_step).
%     w      : n-by-1 IRLS weights p.*(1-p) at the solution (for the Hessian).
%
%   See also hessian_operator, hessian_spectrum, sigmoid, pcg.

    if nargin < 4, opts = struct(); end
    MaxIter = get_opt(opts, 'MaxIter', 100);
    Tol     = get_opt(opts, 'Tol',     1e-8);
    CGtol   = get_opt(opts, 'CGtol',   1e-6);
    CGmaxit = get_opt(opts, 'CGmaxit', 500);
    Verbose = get_opt(opts, 'Verbose', false);

    [~, d] = size(X);
    beta = zeros(d + 1, 1);

    [f, g, p] = logreg_obj(beta, X, y, lambda);
    g0 = max(norm(g), eps);

    obj_trace   = zeros(MaxIter + 1, 1);
    gnorm_trace = zeros(MaxIter + 1, 1);
    cg_per_step = zeros(MaxIter, 1);
    obj_trace(1)   = f;
    gnorm_trace(1) = norm(g);

    converged = false;
    it = 0;
    while it < MaxIter
        if norm(g) <= Tol * g0
            converged = true;
            break;
        end
        it = it + 1;

        w    = p .* (1 - p);
        Hfun = hessian_operator(X, w, lambda);
        [delta, ~, ~, cgIt] = pcg(Hfun, -g, CGtol, CGmaxit);
        cg_per_step(it) = cgIt;

        gd = g' * delta;
        if ~(gd < 0)                          % guard non-descent CG output
            delta = -g;
            gd = g' * delta;
        end

        % Armijo backtracking line search on the penalized objective.
        t = 1; c1 = 1e-4; fnew = logreg_obj(beta + t * delta, X, y, lambda);
        while fnew > f + c1 * t * gd && t > 1e-10
            t = 0.5 * t;
            fnew = logreg_obj(beta + t * delta, X, y, lambda);
        end

        beta = beta + t * delta;
        [f, g, p] = logreg_obj(beta, X, y, lambda);
        obj_trace(it + 1)   = f;
        gnorm_trace(it + 1) = norm(g);

        if Verbose
            fprintf('  newton %2d | f=%.6e | |g|=%.3e | cg=%d | t=%.2g\n', ...
                    it, f, norm(g), cgIt, t);
        end
    end

    w = p .* (1 - p);                          % weights at the returned beta
    yhat = double(p >= 0.5);
    train_acc = mean(yhat == y);

    info = struct( ...
        'newton_iters',      it, ...
        'cg_iters_total',    sum(cg_per_step(1:it)), ...
        'converged',         converged, ...
        'train_acc',         train_acc, ...
        'separable',         train_acc >= 1 - eps, ...
        'obj_trace',         obj_trace(1:it + 1), ...
        'gnorm_trace',       gnorm_trace(1:it + 1), ...
        'cg_iters_per_step', cg_per_step(1:it));
end

%% --------- local helpers ---------
function [f, g, p] = logreg_obj(beta, X, y, lambda)
%LOGREG_OBJ  Penalized logistic objective, gradient, and probabilities.
    d   = numel(beta) - 1;
    wt  = beta(1:d);
    b   = beta(end);
    eta = X * wt + b;
    p   = sigmoid(eta);

    % Stable negative log-likelihood: softplus(eta) - y.*eta.
    ll = softplus(eta) - y .* eta;
    f  = sum(ll) + 0.5 * lambda * (beta' * beta);

    if nargout > 1
        r = p - y;                              % n-by-1 residual
        g = [X' * r + lambda * wt; sum(r) + lambda * b];
    end
end

function s = softplus(z)
%SOFTPLUS  Numerically stable log(1 + exp(z)).
    s = max(z, 0) + log1p(exp(-abs(z)));
end

function v = get_opt(opts, name, default)
%GET_OPT  Fetch opts.(name) with a default.
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end
