function V = varvisc_schur_build_sketch_V(Aapply, n, k, q)
%VARVISC_SCHUR_BUILD_SKETCH_V  Gaussian power sketch of a symmetric operator.
%   AAPPLY may apply either S (largest-tail capture) or S^{-1}
%   (smallest-tail capture).

    validateattributes(k, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(q, {'numeric'}, {'scalar','integer','nonnegative'});
    if ~isa(Aapply, 'function_handle')
        error('varvisc_schur_build_sketch_V:badHandle', 'Aapply must be a handle.');
    end
    k = min(k,n);
    Y = src.precond.subspace_iter_plain(Aapply, randn(n,k), q);
    % This is the sole numerical-range selection in the Schur deflation
    % path.  Every column returned by orth is passed to deflation_P_apply;
    % do not add orth_trunc, an SVD cutoff, or requested-rank slicing later.
    V = orth(real(Y));
    if isempty(V)
        error('varvisc_schur_build_sketch_V:emptyBasis', ...
              'Sketch collapsed to rank zero.');
    end
end
