function [Papply, E, info] = coarse_correction(V, Ahat, tau, kind, output_type)
%COARSE_CORRECTION  The two-level coarse operator, in the form the family needs.
%
%   [Papply, E, info] = COARSE_CORRECTION(V, Ahat, tau, kind, output_type)
%
%   ONE call site for a choice that is made differently for SPD and for
%   symmetric indefinite systems, and that this study previously made the same
%   way for both.
%
%   kind = 'spd'   (SPD Ahat -- the 'ichol' family)
%
%       P = (I - VV') + tau * V (V' Ahat V)^{-1} V'
%
%     built on Ahat DIRECTLY, via src.precond.deflation_P_apply (chol of the
%     coarse matrix).  This is the construction of the SPD reference path,
%     Preconditioner_Recycle/report/ball_surface/solve_deflate_M_P_surface.m and
%     +src/+solver/solve_deflate_M_P.m.  On an exactly invariant span(V) with
%     Ahat V = V L, L > 0, one gets P*Ahat|_V = tau*L^{-1}L = tau*I: the captured
%     modes land on tau.
%
%   kind = 'indef' (symmetric indefinite Ahat -- the 'ildl' family)
%
%       G = (I - VV') + sqrt(tau) * V (V' Ahat^2 V)^{-1/2} V'
%
%     built on the SQUARED operator, via src.precond.deflation_Psqrt_apply.  The
%     square is not optional: E = V'Ahat V is indefinite when Ahat is, and both
%     deflation_P_apply and deflation_Psqrt_apply hard-error on a non-SPD coarse
%     matrix.  Ahat^2 is PSD, so E2 = V'Ahat^2 V > 0 with no |.|-of-eigenvalues
%     trick; the half power then restores the first-power scaling, since
%     (V'Ahat^2 V)^{-1/2} -> |L|^{-1} on an invariant span.  Captured modes land
%     on sqrt(tau)*sign(lambda).
%
%   WHY THIS MATTERS, given that the two agree on an invariant subspace.  There,
%   (V'Ahat^2 V)^{-1/2} = |L|^{-1} = (V'Ahat V)^{-1} for SPD Ahat, so the two
%   forms differ only by tau <-> sqrt(tau) -- a reparametrization, not an error.
%   They part company OFF invariance, which is the regime this whole study is
%   about.  For a one-dimensional coarse space at angle theta from the target of
%   a 2x2 diag(l1,l2), 0 < l1 << l2, the coarse matrix is
%
%       spd    E  = l1 c^2 + l2 s^2      -- l1-dominated until tan^2(theta) > l1/l2
%       indef  E2 = l1^2c^2 + l2^2s^2    -- l1-dominated until tan (theta) > l1/l2
%
%   so the SPD form tolerates misalignment up to sqrt(l1/l2) and the indefinite
%   form only up to l1/l2.  Squaring buys definiteness and costs a square root of
%   angular tolerance.  Measured in exp6.
%
%   INPUTS
%     V            n-by-k basis.  Orthonormal columns are assumed (both backends
%                  assume V'V = I); orthonormalize with orth_trunc first.
%     Ahat         the SPLIT operator, matrix or function handle Afun(X) = Ahat*X.
%     tau          coarse weight, > 0.  Defaults to 0.5, the production value in
%                  both families.
%     kind         'spd' | 'indef'.  Required -- there is deliberately no default,
%                  because silently picking one is the bug this file exists to fix.
%     output_type  'handle' (default) | 'matrix'.  'matrix' needs numeric Ahat.
%
%   OUTPUTS
%     Papply   the apply, or the dense matrix for output_type = 'matrix'
%     E        the k-by-k coarse matrix actually formed (V'Ahat V or V'Ahat^2 V)
%     info     .kind .tau .condE .captured -- where an exactly captured mode
%              lands (tau for 'spd', sqrt(tau) in magnitude for 'indef'), so a
%              caller can report the two families on a common axis.
%
%   See also: two_level_solve_local, deflated_spectrum,
%             src.precond.deflation_P_apply, src.precond.deflation_Psqrt_apply.

    if nargin < 3 || isempty(tau), tau = 0.5; end
    if nargin < 4 || isempty(kind)
        error('coarse_correction:kind', ...
              ['kind is required and must be ''spd'' or ''indef''.  The SPD and ' ...
               'indefinite coarse corrections are different operators; picking ' ...
               'one by default is exactly the mistake this function prevents.']);
    end
    if nargin < 5 || isempty(output_type), output_type = 'handle'; end
    if tau <= 0, error('coarse_correction:tau', 'tau must be positive.'); end

    switch lower(kind)
        case 'spd'
            [Papply, E] = src.precond.deflation_P_apply(V, Ahat, tau, output_type);
            captured    = tau;

        case 'indef'
            [Papply, E] = src.precond.deflation_Psqrt_apply(V, square_op(Ahat), ...
                                                            tau, output_type);
            captured    = sqrt(tau);

        otherwise
            error('coarse_correction:kind', ...
                  'unknown kind "%s" (expected ''spd'' or ''indef'')', kind);
    end

    info          = struct();
    info.kind     = lower(kind);
    info.tau      = tau;
    info.condE    = cond(full(E));
    info.captured = captured;
end

%==========================================================================
function A2 = square_op(Ahat)
%SQUARE_OP  Ahat^2, keeping a handle a handle so no n-by-n matrix is formed.
    if isa(Ahat, 'function_handle')
        A2 = @(Z) Ahat(Ahat(Z));
    else
        A2 = Ahat * Ahat;
    end
end
