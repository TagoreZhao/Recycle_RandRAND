function [Q, r] = orth_trunc(Y)
%ORTH_TRUNC  Orthonormal basis of range(Y) with numerical-rank truncation.
%
%   [Q, r] = ORTH_TRUNC(Y)
%
%   Column-pivoted economy QR (the 3-output form), truncated at the numerical
%   rank.  Unpivoted qr(Y,0) must NOT be used here: its diag(R) is not a rank
%   indicator, so a dependent column in the middle of Y leaves a garbage
%   direction inside the kept block.
%
%   Dropping dependent columns is load-bearing for this study: every basis built
%   here is handed to src.precond.deflation_Psqrt_apply, which forms E = V'AV and
%   hard-errors if any eig(E) <= 0.  It also makes the disk_static control work —
%   there the coupling never moves, the update generator loses half its rank, and
%   the returned block must shrink rather than carry noise directions.
%
%   Matches the convention in subspace_capture/subspace_capture_directed.m's
%   local_orth so capture metrics and deflation bases agree on what "rank" means.
%
%   See also: transport_V, lowrank_update_basis.

    Y = real(full(Y));
    if isempty(Y)
        Q = zeros(size(Y, 1), 0);
        r = 0;
        return;
    end

    [Q0, R, ~] = qr(Y, 0);
    d   = abs(diag(R));
    tol = max(size(Y)) * eps(max(d));
    r   = sum(d > tol);
    Q   = Q0(:, 1:r);
end
