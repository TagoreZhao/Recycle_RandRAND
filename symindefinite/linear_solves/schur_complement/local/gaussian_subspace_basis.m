function [V, info] = gaussian_subspace_basis( ...
        sketchApply, n, k, q, oversampling, reorthogonalize)
%GAUSSIAN_SUBSPACE_BASIS  Return the full oversampled Gaussian sketch basis.
%   By default the sampled subspace is orthogonalized once after q operator
%   products. An optional true REORTHOGONALIZE stabilizes every product for
%   strongly scaled operators. No Rayleigh--Ritz projection, rotation,
%   selection, or truncation is used.

    if nargin < 6, reorthogonalize = false; end

    if ~isa(sketchApply,'function_handle')
        error('gaussian_subspace_basis:badHandle', ...
              'sketchApply must be a function handle.');
    end
    validateattributes(n,{'numeric'},{'scalar','integer','positive'});
    validateattributes(k,{'numeric'},{'scalar','integer','positive','<=',n});
    validateattributes(q,{'numeric'},{'scalar','integer','nonnegative'});
    validateattributes(oversampling,{'numeric'}, ...
        {'scalar','real','finite','>=',1});
    validateattributes(reorthogonalize,{'logical','numeric'}, ...
        {'scalar'},mfilename,'reorthogonalize');

    sketchWidth = min(n,ceil(oversampling*k));
    sample = randn(n,sketchWidth);
    if reorthogonalize
        for powerStep = 1:q
            sample = orth(real(sketchApply(sample)));
            if size(sample,2) ~= sketchWidth
                error('gaussian_subspace_basis:intermediateRankLoss', ...
                    ['Sketch rank %d differs from requested width %d ', ...
                     'after power product %d.'], ...
                    size(sample,2),sketchWidth,powerStep);
            end
        end
    else
        sample = src.precond.subspace_iter_plain(sketchApply,sample,q);
    end
    V = orth(real(sample));
    if size(V,2) ~= sketchWidth
        error('gaussian_subspace_basis:rankLoss', ...
              'Final sketch rank %d differs from requested width %d.', ...
              size(V,2),sketchWidth);
    end

    info = struct('nominalRank',k, ...
                  'oversampling',oversampling, ...
                  'reorthogonalized',logical(reorthogonalize), ...
                  'sketchWidth',sketchWidth, ...
                  'realizedSketchRank',size(V,2), ...
                  'orthogonalityResidual', ...
                  norm(V'*V-eye(sketchWidth),'fro'));
end
