function [tau, info] = amg_sketch_tau(V, Afun)
%AMG_SKETCH_TAU  Deflation shift estimated from the sketched basis alone.
%
%   [tau, info] = amg_sketch_tau(V, Afun) returns
%       tau = lam_max(V' * A * V),      V orthonormal,
%   the largest Ritz value of A on span(V).
%
%   Why this estimator.  The deflation operator
%       P = (I - VV') + tau * V (V'AV)^{-1} V'
%   places every deflated direction at exactly tau, so tau IS lam_min of the
%   preconditioned operator whenever the deflated cluster is the bottom of the
%   spectrum.  The ideal choice is lam_{r+1}(A) -- the first eigenvalue NOT
%   deflated -- which requires an eigensolve.  When span(V) has captured the r
%   smallest eigenvectors, its top Ritz value approximates lam_r from below and
%   therefore sits just under lam_{r+1}: close enough to be the right shift,
%   and computable from the sketch alone.
%
%   That last point is the reason this function exists.  Taking tau from a
%   cached eigendecomposition would hand the deflation arms spectral
%   information the AMG-as-preconditioner arm never gets, which would make the
%   head-to-head unfair.  Everything here comes from V and A matvecs.
%
%   Because tau sets lam_min directly, a mis-estimated tau shows up as a bad
%   condition number that has nothing to do with subspace quality.  Callers
%   should record info.tau alongside the reference lam_{r+1} (where a cache
%   happens to exist) so the two effects stay separable -- see the
%   tau_ritz_gap column in run_amg_deflation_vs_precond.
%
%   Inputs
%     V    : n-by-r basis, assumed orthonormal (from amg_sketch_basis)
%     Afun : @(X) A*X block-capable apply, or the sparse matrix A itself
%
%   Outputs
%     tau  : positive scalar shift (NaN if the Ritz matrix is not positive)
%     info : struct with .tau, .ritz (all r Ritz values, ascending),
%            .ritz_min, .r, .ok, .err
%
%   See also AMG_SKETCH_BASIS, DEFLATION_P_APPLY.

    if isempty(V)
        error('amg_sketch_tau:emptyV', 'V must have at least one column.');
    end
    r = size(V, 2);

    if isa(Afun, 'function_handle')
        AV = Afun(V);
    else
        AV = Afun * V;
    end

    E = V' * AV;
    E = (E + E') / 2;                        % symmetrize the Ritz matrix

    ritz = sort(real(eig(E)), 'ascend');
    tau  = ritz(end);

    info = struct('tau', tau, 'ritz', ritz, 'ritz_min', ritz(1), 'r', r, ...
                  'ok', true, 'err', '');

    if ~(isfinite(tau) && tau > 0)
        % deflation_P_apply rejects non-positive tau; report rather than let
        % the caller hit an error deep inside the projector build.
        info.ok  = false;
        info.err = sprintf('non-positive or non-finite tau (%g)', tau);
        tau      = NaN;
        info.tau = NaN;
    end
end
