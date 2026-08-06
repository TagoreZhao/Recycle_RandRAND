function V = build_sketch_V(Ainv, n, k, q)
%BUILD_SKETCH_V  Gaussian sketch of an INVERSE operator -> coarse space.
%   V = BUILD_SKETCH_V(AINV, N, K, Q)
%
%   Returns an orthonormal basis for the K smallest-eigenvalue modes of an SPD
%   operator A, obtained by Q rounds of block power iteration on its INVERSE
%   applied to a Gaussian test matrix.  Because A^{-1} maps the smallest modes
%   of A to its own largest ones, plain power iteration on the inverse converges
%   to exactly the cluster that stalls PCG -- no eigensolve, no polynomial
%   filter, no spectral bounds needed.
%
%   Inputs
%     AINV     function handle X -> A^{-1} * X, handling a BLOCK X.  In this
%              study A is the Schur complement S ITSELF (there is no split
%              factor), so with the frozen Cholesky factor Rf of S_1 the handle
%              is  @(X) Rf' \ (Rf \ X).
%     N        row dimension of the operator.
%     K        number of sketch columns (= the requested coarse-space width).
%     Q        power-iteration rounds.
%
%   Output
%     V        N-by-M orthonormal basis, M <= K.  M < K means `orth` found the
%              sketch numerically rank deficient; callers must record M rather
%              than assume K (see solve_schur_sequence's deflat_dim).
%
%   NO RE-ORTHOGONALIZATION INSIDE THE LOOP -- src.precond.subspace_iter_plain is
%   used deliberately (never subspace_iter), matching the indefinite sibling's
%   'gaussian' method in src.precond.build_deflation_V.  A single SVD-based
%   `orth` at the end supplies the orthonormality that deflation_P_apply needs
%   and drops the columns the un-reorthogonalized iteration collapsed together;
%   an unpivoted qr(...,0) would keep them and hand chol() a singular V'*Ahat*V.
%
%   LOCAL helper, not promoted to +src: it is the unsplit counterpart of
%   src.precond.build_deflation_V's split-factor form and stays here until the
%   benchmark has validated it.
%
%   See also: src.precond.subspace_iter_plain, src.precond.deflation_P_apply,
%             src.precond.build_deflation_V, solve_schur_sequence.

    if ~isa(Ainv, 'function_handle')
        error('build_sketch_V:badHandle', 'AINV must be a function handle.');
    end
    if ~(isscalar(k) && k == floor(k) && k >= 1)
        error('build_sketch_V:badK', 'K must be a positive integer (got %g).', k);
    end
    if ~(isscalar(q) && q == floor(q) && q >= 0)
        error('build_sketch_V:badQ', 'Q must be a non-negative integer (got %g).', q);
    end
    k = min(k, n);

    Y = src.precond.subspace_iter_plain(Ainv, randn(n, k), q);
    V = orth(real(Y));

    if isempty(V)
        error('build_sketch_V:emptyBasis', ...
              'the sketch collapsed to rank 0 (n=%d, k=%d, q=%d).', n, k, q);
    end
end
