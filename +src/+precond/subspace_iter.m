function V = subspace_iter(apply, Omega, q)
%SUBSPACE_ITER_PLAIN  Plain power iteration with re-orthogonalization.
%   V = SUBSPACE_ITER_PLAIN(apply, Omega, q) applies operator handle
%   'apply' to the starting matrix Omega exactly q times.
    V = orth(full(Omega));          % full() guards against sparse input (e.g. empty-column sparse)
    for i = 1:q
        V = orth(full(apply(V)));   % apply(V) ~ A^{-1}V
    end
end
