function [V, theta, info] = gaussian_rayleigh_ritz_basis( ...
        sketchApply, rayleighApply, n, targetRank, q, oversampling, selection)
%GAUSSIAN_RAYLEIGH_RITZ_BASIS  Oversampled Gaussian sketch, then compress.
%   No orthogonalization is performed between subspace-iteration products.
%   The sample is orthogonalized once at the end, and Rayleigh--Ritz then
%   compresses it to TARGETRANK smallest or largest Ritz vectors.

    if ~isa(sketchApply,'function_handle') || ~isa(rayleighApply,'function_handle')
        error('gaussian_rayleigh_ritz_basis:badHandle', ...
              'Both operators must be function handles.');
    end
    validateattributes(n, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(targetRank, {'numeric'}, ...
        {'scalar','integer','positive','<=',n});
    validateattributes(q, {'numeric'}, {'scalar','integer','nonnegative'});
    validateattributes(oversampling, {'numeric'}, ...
        {'scalar','real','finite','>=',1});
    if ~ismember(selection,{'smallest','largest'})
        error('gaussian_rayleigh_ritz_basis:badSelection', ...
              'selection must be ''smallest'' or ''largest''.');
    end

    sketchWidth = min(n,max(targetRank,ceil(oversampling*targetRank)));
    Y = src.precond.subspace_iter_plain( ...
        sketchApply,randn(n,sketchWidth),q);
    Q = orth(real(Y));
    if size(Q,2) < targetRank
        error('gaussian_rayleigh_ritz_basis:rankLoss', ...
              'Final sketch rank %d is smaller than target rank %d.', ...
              size(Q,2),targetRank);
    end

    projected = Q'*rayleighApply(Q);
    projected = (projected+projected')/2;
    [rotation,values] = eig(projected);
    thetaAll = real(diag(values));
    if strcmp(selection,'smallest')
        [thetaAll,order] = sort(thetaAll,'ascend');
    else
        [thetaAll,order] = sort(thetaAll,'descend');
    end
    selected = order(1:targetRank);
    V = Q*rotation(:,selected);
    theta = thetaAll(1:targetRank);
    info = struct('targetRank',targetRank, ...
                  'sketchWidth',sketchWidth, ...
                  'realizedSketchRank',size(Q,2), ...
                  'orthogonalityResidual', ...
                  norm(V'*V-eye(targetRank),'fro'));
end
