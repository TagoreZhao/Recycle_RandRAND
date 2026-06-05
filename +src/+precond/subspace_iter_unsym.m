function V = subspace_iter_unsym(apply, applyT, Omega, q)
    V = orth(full(Omega));
    for i = 1:q
        V = orth(full(applyT(apply(V))));
    end
end