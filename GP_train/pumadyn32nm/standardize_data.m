function [Xs, ys, stats] = standardize_data(X, y)
%STANDARDIZE_DATA  Standardize features and target to zero mean, unit variance.
%   [Xs, ys, stats] = STANDARDIZE_DATA(X, y) returns standardized copies of
%   the feature matrix X and target vector y. Each feature column of X and
%   the target y are shifted to zero mean and scaled to unit standard
%   deviation. Inputs are not modified (new arrays are returned).
%
%   Inputs
%     X      : n-by-d feature matrix.
%     y      : n-by-1 target vector.
%
%   Outputs
%     Xs     : n-by-d standardized features.
%     ys     : n-by-1 standardized target.
%     stats  : struct with fields mu_X (1-by-d), sigma_X (1-by-d),
%              mu_y (scalar), sigma_y (scalar) used for the transform.
%
%   Example
%     [Xs, ys, stats] = standardize_data(X, y);
%
%   Implementation notes
%     - Columns (or y) with (near) zero variance are scaled by 1 instead of 0
%       to avoid division by zero; such columns become all-zero after centering.

    if size(X, 1) ~= numel(y)
        error('standardize_data:sizeMismatch', ...
              'X has %d rows but y has %d elements.', size(X, 1), numel(y));
    end

    mu_X    = mean(X, 1);
    sigma_X = std(X, 0, 1);
    sigma_X(sigma_X < eps) = 1;          % guard zero-variance columns
    Xs = (X - mu_X) ./ sigma_X;

    mu_y    = mean(y);
    sigma_y = std(y, 0);
    if sigma_y < eps                     % guard zero-variance target
        sigma_y = 1;
    end
    ys = (y - mu_y) ./ sigma_y;

    stats = struct('mu_X', mu_X, 'sigma_X', sigma_X, ...
                   'mu_y', mu_y, 'sigma_y', sigma_y);
end
