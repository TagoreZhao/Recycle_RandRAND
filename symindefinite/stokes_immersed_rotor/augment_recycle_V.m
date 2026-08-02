function V = augment_recycle_V(Vbase, W)
%AUGMENT_RECYCLE_V  Append a recycled Krylov block to a deflation coarse basis.
%
%   V = AUGMENT_RECYCLE_V(VBASE, W) returns an orthonormal basis spanning
%   range(VBASE) + range(W), where VBASE is an already-orthonormal deflation basis
%   and W is the RAW block of ILDL-preconditioned residuals harvested from the
%   previous solve (see make_recording_pdef).
%
%   Three steps, in this order:
%     1. scale each column of W to unit 2-norm — MINRES residuals span many orders
%        of magnitude, and the rank test in step 3 is relative to the largest column;
%     2. project off range(VBASE) — VBASE is orthonormal, so I - VBASE*VBASE' is the
%        exact orthogonal projector and [VBASE, orth(W_perp)] is orthonormal by
%        construction, with no need to re-orthogonalize the (large) base block;
%     3. orth() the remainder — this is where numerically dependent recycled columns
%        are dropped, which is what keeps V'*Ahat^2*V safely SPD for
%        deflation_Psqrt_apply.
%
%   Following the SPD/PCG prior art in
%   Preconditioner_Recycle/report/ball_surface_krylov_recycle, the harvested columns
%   are carried RAW and orthogonalization is applied ONLY here, at the point the
%   block becomes a deflation basis.
%
%   Inputs:
%     Vbase - n x kb orthonormal deflation basis ([] is allowed)
%     W     - n x kw raw recycled block ([] or 0 columns -> V = Vbase)
%
%   Output:
%     V - n x (kb + kr) orthonormal basis, kr <= kw
%
%   LOCAL trial version; promotable to +src/+precond once validated.
%
%   See also: make_recording_pdef, define_solver_list,
%             src.precond.deflation_Psqrt_apply.

    if isempty(W) || size(W, 2) == 0
        V = Vbase;
        return;
    end
    if isempty(Vbase)
        V = orth(W ./ max(vecnorm(W), eps));
        return;
    end

    W = W ./ max(vecnorm(W), eps);       % unit columns
    W = W - Vbase * (Vbase' * W);        % project off the base space
    Wr = orth(W);                        % orth ONLY here; drops dependent columns
    V = [Vbase, Wr];
end
