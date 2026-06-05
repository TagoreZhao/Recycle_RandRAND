function [Xs, scale, keep] = standardize_features(X)
%STANDARDIZE_FEATURES  Sparse-safe column scaling for high-dimensional data.
%   [Xs, scale, keep] = STANDARDIZE_FEATURES(X) drops all-zero feature columns
%   and rescales every remaining column to unit root-mean-square, WITHOUT
%   mean-centering. Centering is intentionally skipped: subtracting column
%   means would destroy sparsity and densify ~10^4-10^5 column matrices such
%   as rcv1/real-sim. This is a deliberate deviation from the full z-score used
%   in GP_train/standardize_data.m, required at this scale.
%
%   Input
%     X     : n-by-d (sparse) feature matrix.
%   Outputs
%     Xs    : n-by-numel(keep) sparse matrix, each column unit RMS.
%     scale : 1-by-numel(keep) RMS scale factors applied (per kept column).
%     keep  : indices (into 1:d) of the columns retained (non-all-zero).
%
%   Implementation notes
%     - Columns with (near) zero energy are dropped; constant/all-zero columns
%       would otherwise inject exact-zero data directions into the Hessian.
%     - Scaling is applied via a sparse diagonal so the result stays sparse.

    [n, d] = size(X);
    ss = full(sum(X.^2, 1));                 % 1-by-d column energies
    keep = find(ss > 0);
    if isempty(keep)
        error('standardize_features:allZero', 'All feature columns are zero.');
    end

    Xk  = X(:, keep);
    rms = sqrt(full(sum(Xk.^2, 1)) / n);     % 1-by-numel(keep)
    rms(rms < eps) = 1;                      % guard (should not trigger after drop)

    nk    = numel(keep);
    Dinv  = spdiags((1 ./ rms(:)), 0, nk, nk);
    Xs    = Xk * Dinv;                       % stays sparse
    scale = rms;

    if numel(keep) < d
        fprintf('standardize_features: dropped %d/%d all-zero columns.\n', ...
                d - numel(keep), d);
    end
end
