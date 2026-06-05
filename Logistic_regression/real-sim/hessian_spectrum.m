function spec = hessian_spectrum(X, w, lambda, opts)
%HESSIAN_SPECTRUM  Eigenvalues of the logistic Newton Hessian H = Xa'WXa+lambda*I.
%   spec = HESSIAN_SPECTRUM(X, w, lambda, opts) computes the smallest-K and
%   largest-K eigenvalues of the (d+1)-by-(d+1) penalized Hessian at the IRLS
%   weights w, mirroring GP_train/plot_kernel_spectrum.m. Two regimes:
%
%     dim = d+1 <= DenseMax : form H densely, take the FULL spectrum, and slice
%                             both ends (definitive small-eigenvalue spike).
%     dim > DenseMax        : matrix-free eigs on the operator handle. The
%                             largest end uses the forward handle with
%                             'largestabs'; the smallest end uses the INVERSE
%                             handle Hinv = @(x) H\x (an inner PCG solve) with
%                             'smallestabs', because eigs in shift-invert mode
%                             requires the function handle to return A\x.
%
%   Inputs
%     X      : n-by-d (sparse) standardized features.
%     w      : n-by-1 IRLS weights p.*(1-p).
%     lambda : L2 penalty (the spectrum floor).
%     opts   : struct with fields DenseMax (8000) and K (500).
%   Output
%     spec   : struct with fields
%                eigs_small (smallest-K, ascending),
%                eigs_large (largest-K, descending),
%                mode ("full"|"topk"), min, max, cond, lambda_floor,
%                n_at_floor (count within 1.01*lambda; spike multiplicity).
%
%   Performance: for the matrix-free regime (e.g. rcv1, d=47236) the small end
%   is a shift-invert eigs that solves H\x by inner PCG on every Lanczos step;
%   computing 500 smallest eigenvalues there is expensive but matrix-free.
%
%   See also hessian_operator, logreg_newton, plot_hessian_spectrum.

    if nargin < 4, opts = struct(); end
    DenseMax = get_opt(opts, 'DenseMax', 8000);
    K        = get_opt(opts, 'K',        500);

    [n, d] = size(X);
    dim = d + 1;
    ks  = min(K, floor(dim / 2));           % per-end count; ends never overlap

    if dim <= DenseMax
        % Dense H = [X 1]' diag(w) [X 1] + lambda*I; full spectrum, slice ends.
        Xa = [X, ones(n, 1)];
        B  = sqrt(w) .* Xa;                 % row-scaled; stays sparse
        H  = full(B' * B) + lambda * eye(dim);
        H  = (H + H') / 2;
        e  = sort(eig(H), 'ascend');
        eigs_small = e(1:ks);                       % ascending
        eigs_large = e(end:-1:end - ks + 1);        % descending (largest first)
        mode = "full";
        eMin = e(1);  eMax = e(end);  nFloor = sum(e <= 1.01 * lambda);
    else
        Hfun = hessian_operator(X, w, lambda);
        eigsOpts = struct('IsFunctionSymmetric', true, ...
                          'Tolerance', 1e-8, 'MaxIterations', 5000);

        % Largest end: forward handle.
        eL = eigs(Hfun, dim, ks, 'largestabs', eigsOpts);
        eigs_large = sort(real(eL), 'descend');

        % Smallest end: inverse handle. eigs('smallestabs') with a function
        % handle is shift-invert and requires the handle to return H\x.
        Hinv = @(x) cg_apply(Hfun, x, 1e-8, min(dim, 2000));
        eS = eigs(Hinv, dim, ks, 'smallestabs', eigsOpts);
        eigs_small = sort(real(eS), 'ascend');

        mode = "topk";
        eMin = eigs_small(1);  eMax = eigs_large(1);
        nFloor = sum(eigs_small <= 1.01 * lambda);
        fprintf(['hessian_spectrum: dim=%d > DenseMax=%d; matrix-free ' ...
                 'smallest-%d (inverse-handle PCG) and largest-%d.\n'], ...
                dim, DenseMax, ks, ks);
    end

    spec = struct( ...
        'eigs_small',   eigs_small, ...
        'eigs_large',   eigs_large, ...
        'mode',         mode, ...
        'min',          eMin, ...
        'max',          eMax, ...
        'cond',         eMax / eMin, ...
        'lambda_floor', lambda, ...
        'n_at_floor',   nFloor);
end

%% --------- local helpers ---------
function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end

function y = cg_apply(Hfun, x, tol, maxit)
%CG_APPLY  Approximate H\x by matrix-free PCG (the inverse handle for eigs).
%   eigs calls this once per Lanczos step; the convergence flag is captured to
%   keep the solver silent rather than printing a warning on each call.
    [y, ~] = pcg(Hfun, x, tol, maxit);
end
