function K = ard_rbf_cross_kernel(Xa, Xb, ell)
%ARD_RBF_CROSS_KERNEL  ARD RBF covariance between two point sets.

    if size(Xa, 2) ~= size(Xb, 2)
        error('ard_rbf_cross_kernel:featureMismatch', ...
              'Xa and Xb must have the same number of feature columns.');
    end
    ell = ell(:).';
    d = size(Xa, 2);
    if numel(ell) ~= d || any(~isfinite(ell)) || any(ell <= 0)
        error('ard_rbf_cross_kernel:badLengthscales', ...
              'ell must contain one positive finite value per feature.');
    end

    exponent = zeros(size(Xa, 1), size(Xb, 1));
    for r = 1:d
        exponent = exponent + (Xa(:, r) - Xb(:, r).').^2 / ell(r)^2;
    end
    K = exp(-0.5 * exponent);
end
