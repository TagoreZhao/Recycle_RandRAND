function V = subspace_iter_plain(apply, Omega, q)
%SUBSPACE_ITER_PLAIN  Plain power iteration without re-orthogonalization.
%   V = SUBSPACE_ITER_PLAIN(apply, Omega, q) applies operator handle
%   'apply' to the starting matrix Omega exactly q times.

    V = Omega;
    for i = 1:q
        V = apply(V);
    end
end
