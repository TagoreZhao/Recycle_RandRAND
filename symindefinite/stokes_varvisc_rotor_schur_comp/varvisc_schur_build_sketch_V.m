function V = varvisc_schur_build_sketch_V(Ainv, n, k, q)
%VARVISC_SCHUR_BUILD_SKETCH_V  Gaussian inverse-power sketch of an SPD matrix.

    validateattributes(k, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(q, {'numeric'}, {'scalar','integer','nonnegative'});
    if ~isa(Ainv, 'function_handle')
        error('varvisc_schur_build_sketch_V:badHandle', 'Ainv must be a handle.');
    end
    k = min(k,n);
    Y = src.precond.subspace_iter_plain(Ainv, randn(n,k), q);
    V = orth(real(Y));
    if isempty(V)
        error('varvisc_schur_build_sketch_V:emptyBasis', ...
              'Sketch collapsed to rank zero.');
    end
end
