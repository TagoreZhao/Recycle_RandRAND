function K = rbf_kernel_matrix(X, ell)
%RBF_KERNEL_MATRIX  Dense RBF (squared-exponential) kernel matrix.
%   K = RBF_KERNEL_MATRIX(X, ELL) returns the n-by-n kernel matrix
%
%       K(i,k) = exp( -||X(i,:) - X(k,:)||^2 / (2*ELL^2) ),
%
%   for the n rows of X and lengthscale ELL.
%
%   Inputs
%     X    : n-by-d data matrix.
%     ell  : positive scalar lengthscale.
%
%   Output
%     K    : n-by-n dense symmetric kernel matrix (K(i,i) = 1).
%
%   Example
%     K = rbf_kernel_matrix(X, 1.5);
%     A = K + 1e-3 * eye(size(K, 1));   % SPD kernel-ridge system matrix
%
%   Implementation notes
%     - Dense by design; memory scales as O(n^2).

    if ~(isscalar(ell) && ell > 0 && isfinite(ell))
        error('rbf_kernel_matrix:badLengthscale', ...
              'ell must be a positive finite scalar.');
    end

    D2 = pairwise_sq_dist(X);
    K = exp(-D2 / (2 * ell^2));
    K = (K + K.') / 2;                  % enforce exact symmetry
end
