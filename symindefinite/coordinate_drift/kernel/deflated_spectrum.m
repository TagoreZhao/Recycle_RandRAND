function out = deflated_spectrum(Ahat, V, tau, kind)
%DEFLATED_SPECTRUM  Spectrum of the two-level preconditioned split operator.
%
%   out = DEFLATED_SPECTRUM(Ahat, V, tau)          % kind = 'indef' (default)
%   out = DEFLATED_SPECTRUM(Ahat, V, tau, kind)
%
%   Forms the coarse correction its family actually runs -- via
%   coarse_correction(V, Ahat, tau, kind, 'matrix') -- and returns the spectrum
%   of the operator the Krylov method sees,
%
%       B = G * Ahat        (similar to the symmetric G^{1/2} Ahat G^{1/2}).
%
%   kind = 'indef'  (symmetric indefinite Ahat, the 'ildl' family; MINRES)
%
%       G = (I - VV') + sqrt(tau) * V (V' Ahat^2 V)^{-1/2} V'
%
%     built on the SQUARED operator, because V'Ahat V is indefinite when Ahat is
%     and the coarse solve would break.  On an exactly captured mode G*Ahat has
%     eigenvalue sqrt(tau)*sign(lambda) -- the textbook deflation target, and the
%     reason tau = 0.5 is a sensible O(1) choice.
%
%   kind = 'spd'    (SPD Ahat, the 'ichol' family; PCG)
%
%       G = (I - VV') + tau * V (V' Ahat V)^{-1} V'
%
%     built on Ahat DIRECTLY -- the construction of the SPD reference path
%     (ball_surface / +src/+solver/solve_deflate_M_P.m).  On an exactly captured
%     mode G*Ahat has eigenvalue tau.  No squaring, so the coarse matrix is
%     conditioned like Ahat rather than like Ahat^2.
%
%   G, not G*Ahat*G:  MATLAB's minres and pcg take their fifth argument as the
%   preconditioner M and apply M^{-1}, and a FUNCTION HANDLE there is the apply
%   of M^{-1} itself.  two_level_solve_local passes the handle that applies G, so
%   M^{-1} = G and the preconditioned operator is G*Ahat.  The distinction is not
%   cosmetic: G*Ahat*G would give tau/lambda, which would blow up precisely on
%   the modes being deflated.
%
%   Everything returned is a function of span(V) only: replacing V by V*Q with
%   Q orthogonal leaves G -- and hence every field below -- unchanged, in either
%   form.  (Both proofs are the same: (Q'EQ)^{-p} = Q'E^{-p}Q for p = 1/2 or 1.)
%
%   out fields:
%     .lam        eigenvalues of B, ascending.  Computed as eig(G^1/2 Ahat G^1/2),
%                 which is symmetric and similar to G*Ahat, so the spectrum is
%                 that of G*Ahat exactly (pinned by tests T17, T20).
%     .kappa      max|lam| / min|lam|, the quantity the Krylov method pays for
%     .lam_min_abs, .lam_max_abs
%     .minres_rate  the standard INDEFINITE bound's asymptotic factor,
%                   ((sqrt(a*d)-sqrt(b*c))/(sqrt(a*d)+sqrt(b*c)))^{1/2} for the
%                   enclosing intervals [-a,-b] u [c,d]; NaN if definite.
%     .cg_rate      the standard DEFINITE bound's asymptotic factor,
%                   (sqrt(kappa)-1)/(sqrt(kappa)+1); NaN if the spectrum
%                   straddles zero.  Exactly one of the two rates is finite, so
%                   the SPD family is not left without a convergence estimate.
%     .kappa_undeflated  the same ratio for Ahat itself, for reference.
%     .G, .E, .condE, .kind
%
%   Dense by design: this is the small closed-form model of Thm 4.1, not a
%   large-scale path.
%
%   See also: coarse_correction, two_level_solve_local.

    if nargin < 3 || isempty(tau),  tau  = 1;       end
    if nargin < 4 || isempty(kind), kind = 'indef'; end

    Ahat = full((Ahat + Ahat') / 2);
    V    = orth_trunc(V);

    [G, E, cinfo] = coarse_correction(V, Ahat, tau, kind, 'matrix');
    G   = (G + G') / 2;
    Gh  = sqrtm_spd(G);
    B   = Gh * Ahat * Gh;                  % symmetric, similar to G*Ahat
    B   = (B + B') / 2;

    out       = struct();
    out.G     = G;
    out.E     = E;
    out.condE = cinfo.condE;
    out.kind  = cinfo.kind;
    out.lam   = sort(real(eig(B)), 'ascend');
    a         = abs(out.lam);
    out.lam_min_abs = min(a);
    out.lam_max_abs = max(a);
    out.kappa = out.lam_max_abs / max(out.lam_min_abs, realmin);

    la = abs(real(eig(Ahat)));
    out.kappa_undeflated = max(la) / max(min(la), realmin);

    out.minres_rate = minres_rate(out.lam);
    out.cg_rate     = cg_rate(out.lam);
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
%   bound, cg_rate below, and the two-interval formula does not apply).
    neg = lam(lam < 0);
    pos = lam(lam > 0);
    if isempty(neg) || isempty(pos), r = NaN; return; end
    a = max(abs(neg));  b = min(abs(neg));
    d = max(pos);       c = min(pos);
    num = sqrt(a * d) - sqrt(b * c);
    den = sqrt(a * d) + sqrt(b * c);
    r   = sqrt(max(num, 0) / den);
end

function r = cg_rate(lam)
%CG_RATE  Asymptotic factor of the classical definite CG bound.
%   For a one-signed spectrum with condition number kappa, the error decays like
%   r^m with r = (sqrt(kappa) - 1) / (sqrt(kappa) + 1).  Returns NaN when the
%   spectrum straddles zero, where CG does not apply and minres_rate does.
    neg = lam(lam < 0);
    pos = lam(lam > 0);
    if ~isempty(neg) && ~isempty(pos), r = NaN; return; end
    a = abs(lam);
    kap = max(a) / max(min(a), realmin);
    r   = (sqrt(kap) - 1) / (sqrt(kap) + 1);
end
