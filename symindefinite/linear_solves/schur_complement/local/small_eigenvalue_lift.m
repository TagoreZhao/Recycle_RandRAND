function [Papply, PsqrtApply] = small_eigenvalue_lift(V, tau)
%SMALL_EIGENVALUE_LIFT  Lift an orthonormal subspace by a fixed factor.
%   [Papply, PsqrtApply] = SMALL_EIGENVALUE_LIFT(V, tau) returns handles for
%
%       P       = I + tau^(-1) V V',
%       P^(1/2) = I + (sqrt(1+tau^(-1))-1) V V'.
%
%   V must have orthonormal columns and tau must be a positive finite scalar.

    assert(isnumeric(V) && ismatrix(V) && ~isempty(V), ...
        'V must be a nonempty numeric matrix.');
    assert(isscalar(tau) && tau > 0 && isfinite(tau), ...
        'The lifting tau must be a positive finite scalar.');

    Vt = V';
    orthogonalityResidual = norm(Vt*V-eye(size(V,2)), 'fro');
    assert(orthogonalityResidual <= 1e-10, ...
        'V must have orthonormal columns (residual %.3e).', ...
        orthogonalityResidual);

    inverseTau = 1/tau;
    squareRootWeight = sqrt(1+inverseTau)-1;
    Papply = @(X) local_lift_apply(X,V,Vt,inverseTau);
    PsqrtApply = @(X) local_lift_apply(X,V,Vt,squareRootWeight);
end

function Y = local_lift_apply(X,V,Vt,weight)
    Y = X + weight*V*(Vt*X);
end
