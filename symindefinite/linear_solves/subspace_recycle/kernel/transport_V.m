function [Vn, info] = transport_V(U, P, C)
%TRANSPORT_V  Re-express a deflation space in the CURRENT split coordinates.
%
%   [Vn, info] = TRANSPORT_V(U, P)
%   [Vn, info] = TRANSPORT_V(U, P, C)     % pass a precomputed C to skip rebuilding
%
%   U is the deflation space in ORIGINAL coordinates (the M-orthonormal
%   generalized eigenvectors from eigs(K, M, k, 'smallestabs')).  P is the
%   CURRENT step's make_ildl_precond struct.  Returns an orthonormal basis of
%
%       V_n = orth( C_n' * U )
%
%   which is the basis build_deflation_V would have produced at this step had it
%   been given these eigenvectors — i.e. the coordinate-drift fix (H1).
%
%   Why cache U rather than V: build_deflation_V.m:87-89 computes U and then
%   discards it (`Vs = C' * Us`).  Keeping U makes re-expression a sparse matvec
%   block plus a QR — no triangular solve, no round-trip through C_ref^-T, and no
%   accumulated error.  Cost is k spmv + one n-by-k QR per step, with no
%   eigensolve and no factorization.
%
%   To go the other way (recover what a frozen V actually deflates at this step)
%   just apply the handle directly: U_eff = P.applyCtinv(V_frozen).  That is the
%   quantity run_ildl_drift compares against U_ref.
%
%   info: .k_in, .k_out (numerical rank kept), .rank_drop, .time.
%
%   See also: ildl_coordinate_map, src.precond.build_deflation_V.

    t0 = tic;
    if isempty(U)
        Vn   = zeros(size(U, 1), 0);
        info = struct('k_in', 0, 'k_out', 0, 'rank_drop', 0, 'time', toc(t0));
        return;
    end
    if nargin < 3 || isempty(C)
        C = ildl_coordinate_map(P);
    end

    Vn = orth_trunc(C' * U);

    info = struct('k_in', size(U, 2), 'k_out', size(Vn, 2), ...
                  'rank_drop', size(U, 2) - size(Vn, 2), 'time', toc(t0));
end
