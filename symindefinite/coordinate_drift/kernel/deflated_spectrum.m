function out = deflated_spectrum(Ahat, V, tau)
%DEFLATED_SPECTRUM  Spectrum of the two-level preconditioned split operator.
%
%   out = DEFLATED_SPECTRUM(Ahat, V, tau)
%
%   Forms the production coarse correction of two_level_split_solve,
%
%       G = (I - V V') + sqrt(tau) * V (V' Ahat^2 V)^{-1/2} V' ,
%
%   via src.precond.deflation_Psqrt_apply(V, Ahat^2, tau, 'matrix'), and returns
%   the spectrum of the operator MINRES actually sees,
%
%       B = G * Ahat        (similar to the symmetric G^{1/2} Ahat G^{1/2}).
%
%   G, not G*Ahat*G:  MATLAB's minres takes its fifth argument as the
%   preconditioner M and applies M^{-1}, and a FUNCTION HANDLE there is the
%   apply of M^{-1} itself.  two_level_split_solve passes the handle that
%   applies G, so M^{-1} = G and the preconditioned operator is G*Ahat.  The
%   distinction is not cosmetic: on an exactly captured mode G*Ahat has
%   eigenvalue sqrt(tau)*sign(lambda) -- the textbook deflation target, and the
%   reason tau = 0.5 is a sensible O(1) choice -- whereas G*Ahat*G would give
%   tau/lambda, which would blow up precisely on the modes being deflated.
%
%   Everything returned is a function of span(V) only: replacing V by V*Q with
%   Q orthogonal leaves G -- and hence every field below -- unchanged.
%
%   out fields:
%     .lam        eigenvalues of B, ascending.  Computed as eig(G^1/2 Ahat G^1/2),
%                 which is symmetric and similar to G*Ahat, so the spectrum is
%                 that of G*Ahat exactly (pinned by test T17).
%     .kappa      max|lam| / min|lam|, the quantity MINRES pays for
%     .lam_min_abs, .lam_max_abs
%     .minres_rate  the standard indefinite bound's asymptotic factor,
%                   ((sqrt(a*d)-sqrt(b*c))/(sqrt(a*d)+sqrt(b*c)))^{1/2} for the
%                   enclosing intervals [-a,-b] u [c,d]; NaN if definite.
%     .kappa_undeflated  the same ratio for Ahat itself, for reference.
%
%   Dense by design: this is the small closed-form model of Thm 4.1, not a
%   large-scale path.
%
%   See also: src.precond.deflation_Psqrt_apply, src.precond.two_level_split_solve.

    if nargin < 3 || isempty(tau), tau = 1; end

    Ahat = full((Ahat + Ahat') / 2);
    V    = orth_trunc(V);

    G   = src.precond.deflation_Psqrt_apply(V, Ahat * Ahat, tau, 'matrix');
    G   = (G + G') / 2;
    Gh  = sqrtm_spd(G);
    B   = Gh * Ahat * Gh;                  % symmetric, similar to G*Ahat
    B   = (B + B') / 2;

    out       = struct();
    out.G     = G;
    out.lam   = sort(real(eig(B)), 'ascend');
    a         = abs(out.lam);
    out.lam_min_abs = min(a);
    out.lam_max_abs = max(a);
    out.kappa = out.lam_max_abs / max(out.lam_min_abs, realmin);

    la = abs(real(eig(Ahat)));
    out.kappa_undeflated = max(la) / max(min(la), realmin);

    out.minres_rate = minres_rate(out.lam);
end

%==========================================================================
function S = sqrtm_spd(G)
%SQRTM_SPD  Symmetric square root of an SPD matrix, by eigendecomposition.
    [U, D] = eig(full((G + G')/2));
    S = U * diag(sqrt(max(real(diag(D)), 0))) * U';
    S = (S + S') / 2;
end

function r = minres_rate(lam)
%MINRES_RATE  Asymptotic factor of the classical indefinite MINRES bound.
%   For sigma(B) contained in [-a,-b] u [c,d] with 0 < b <= a, 0 < c <= d, the
%   two-interval Chebyshev bound decays like r^m with
%       r = sqrt( (sqrt(a*d) - sqrt(b*c)) / (sqrt(a*d) + sqrt(b*c)) ).
%   Returns NaN when the spectrum is one-signed (the definite case has its own
%   bound and the two-interval formula does not apply).
    neg = lam(lam < 0);
    pos = lam(lam > 0);
    if isempty(neg) || isempty(pos), r = NaN; return; end
    a = max(abs(neg));  b = min(abs(neg));
    d = max(pos);       c = min(pos);
    num = sqrt(a * d) - sqrt(b * c);
    den = sqrt(a * d) + sqrt(b * c);
    r   = sqrt(max(num, 0) / den);
end
