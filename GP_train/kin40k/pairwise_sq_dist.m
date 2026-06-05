function D2 = pairwise_sq_dist(X)
%PAIRWISE_SQ_DIST  Squared Euclidean distances between all rows of X.
%   D2 = PAIRWISE_SQ_DIST(X) returns the n-by-n matrix of squared pairwise
%   distances for the n rows of X, computed via the inner-product identity
%   ||xi - xk||^2 = ||xi||^2 + ||xk||^2 - 2*xi'*xk.
%
%   Inputs
%     X   : n-by-d matrix of row vectors.
%
%   Output
%     D2  : n-by-n symmetric matrix, D2(i,k) = ||X(i,:) - X(k,:)||^2,
%           clamped to be nonnegative (rounding can produce small negatives).
%
%   Example
%     D2 = pairwise_sq_dist(X);
%     D  = sqrt(D2);

    s  = sum(X.^2, 2);                 % n-by-1 squared norms
    G  = X * X.';                      % Gram matrix
    D2 = s + s.' - 2 * G;
    D2 = max(D2, 0);                   % clamp numerical negatives
end
