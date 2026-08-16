function [K, dKlogell, D2parts] = ard_rbf_kernel(X, ell, D2parts)
%ARD_RBF_KERNEL  ARD squared-exponential kernel and log-lengthscale derivatives.
%   K = ARD_RBF_KERNEL(X, ELL) builds
%       K(i,j) = exp(-0.5 * sum_r (X(i,r)-X(j,r))^2 / ELL(r)^2).
%
%   [K, DK, D2PARTS] additionally returns DK{r} = dK/dlog(ELL(r)) and
%   the reusable per-coordinate squared-distance matrices.  Pass D2PARTS
%   back on later calls for the same X to avoid recomputing distances.

    d = size(X, 2);
    ell = ell(:).';
    if numel(ell) ~= d || any(~isfinite(ell)) || any(ell <= 0)
        error('ard_rbf_kernel:badLengthscales', ...
              'ell must contain one positive finite value per feature.');
    end

    if nargin < 3 || isempty(D2parts)
        D2parts = cell(1, d);
        for r = 1:d
            xr = X(:, r);
            D2r = (xr - xr.').^2;
            D2parts{r} = max((D2r + D2r.') / 2, 0);
        end
    elseif ~iscell(D2parts) || numel(D2parts) ~= d
        error('ard_rbf_kernel:badDistances', ...
              'D2parts must be a cell array with one matrix per feature.');
    end

    exponent = zeros(size(X, 1));
    for r = 1:d
        if ~isequal(size(D2parts{r}), [size(X, 1), size(X, 1)])
            error('ard_rbf_kernel:badDistanceSize', ...
                  'D2parts{%d} has the wrong size.', r);
        end
        exponent = exponent + D2parts{r} / ell(r)^2;
    end
    K = exp(-0.5 * exponent);
    K = (K + K.') / 2;
    K(1:size(K, 1)+1:end) = 1;

    if nargout >= 2
        dKlogell = cell(1, d);
        for r = 1:d
            % d/dlog(ell_r) exp(-D_r^2/(2 ell_r^2))
            dKlogell{r} = K .* (D2parts{r} / ell(r)^2);
            dKlogell{r} = (dKlogell{r} + dKlogell{r}.') / 2;
        end
    end
end
