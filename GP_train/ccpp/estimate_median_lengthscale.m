function ell0 = estimate_median_lengthscale(X, maxSamples)
%ESTIMATE_MEDIAN_LENGTHSCALE  Median pairwise distance heuristic for the RBF scale.
%   ell0 = ESTIMATE_MEDIAN_LENGTHSCALE(X) returns the median Euclidean
%   distance between rows of X, using at most 2000 randomly sampled rows.
%
%   ell0 = ESTIMATE_MEDIAN_LENGTHSCALE(X, MAXSAMPLES) uses at most MAXSAMPLES
%   randomly sampled rows.
%
%   Inputs
%     X          : n-by-d data matrix.
%     maxSamples : (optional, default 2000) cap on the number of sampled rows.
%
%   Output
%     ell0       : scalar base lengthscale (median pairwise distance).
%
%   Example
%     ell0 = estimate_median_lengthscale(X);
%
%   Implementation notes
%     - Uses the caller's RNG state; seed externally for reproducibility.
%     - The median is taken over the strict upper triangle (each unordered
%       pair once, excluding the zero diagonal).

    if nargin < 2 || isempty(maxSamples)
        maxSamples = 2000;
    end

    n = size(X, 1);
    if n < 2
        error('estimate_median_lengthscale:tooFewPoints', ...
              'Need at least 2 points to estimate a lengthscale.');
    end

    m = min(maxSamples, n);
    idx = randperm(n, m);
    Xs = X(idx, :);

    D2 = pairwise_sq_dist(Xs);
    mask = triu(true(m), 1);            % strict upper triangle
    dists = sqrt(D2(mask));

    ell0 = median(dists);
    if ~(ell0 > 0 && isfinite(ell0))
        error('estimate_median_lengthscale:degenerate', ...
              'Median distance is non-positive or non-finite (duplicate points?).');
    end
end
