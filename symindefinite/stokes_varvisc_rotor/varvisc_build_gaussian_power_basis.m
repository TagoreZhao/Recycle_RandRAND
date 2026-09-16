function [V, info] = varvisc_build_gaussian_power_basis(C, Omega, q, dK)
%VARVISC_BUILD_GAUSSIAN_POWER_BASIS Explicit paired start, existing q convention.
% dK is an exact LDL decomposition of the CURRENT physical matrix K.
% B = C' K^-1 C; q means exactly q applications, with only a final QR.
    validateattributes(q, {'numeric'}, {'scalar','integer','positive'});
    assert(size(Omega,1)==size(C,1) && size(Omega,2)<=size(C,1));
    Y = src.precond.subspace_iter_plain(@(Z) C'*(dK\(C*Z)), Omega, q);
    [V,R] = qr(real(Y),0);
    % Diagnose rank without changing the established unpivoted-QR basis.
    sigma = svd(R);
    numerical_rank = sum(sigma > max(size(Y))*eps(max(sigma)));
    info = struct('inverse_applications',q, ...
        'inverse_rhs_columns',q*size(Omega,2), ...
        'numerical_rank',numerical_rank,'requested_rank',size(Omega,2));
    if numerical_rank < size(Omega,2)
        warning('varvisc:powerRankLoss','Power sketch numerical rank %d < %d.', ...
            numerical_rank,size(Omega,2));
    end
end
