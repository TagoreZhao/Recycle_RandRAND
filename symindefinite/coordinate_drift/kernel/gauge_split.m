function out = gauge_split(C1, C2, V1)
%GAUGE_SPLIT  Split the chart drift into a metric part and a gauge part.
%
%   out = GAUGE_SPLIT(C1, C2, V1)
%
%   C1, C2 are the step-n and step-(n+1) preconditioner factors (M = C*C'), V1
%   is a basis of the frozen chart-side space.  The transport is
%
%       T = C2' * C1^-T .
%
%   The factor is not determined by M: C and C*Q give the same M for every
%   orthogonal Q.  Write C2 = C2t * Q with Q the Procrustes-optimal alignment
%
%       Q = argmin_{Q in O(N)} || C2 - C1*Q ||_F  =  W*Z',   C1'*C2 = W*S*Z',
%
%   so C2t = C2*Q' is the factor of M2 sitting in C1's gauge.  Then
%
%       T = Q' * Tt,        Tt = C2t' * C1^-T ,
%
%   and with W1 = Tt*V1 the triangle inequality gives
%
%       delta_chart <= delta_metric + delta_gauge,
%       delta_metric = d(span V1, span W1),      (M1 vs M2, gauge held fixed)
%       delta_gauge  = d(span W1, span Q'*W1).   (same M2, different factor)
%
%   delta_metric is the part a better preconditioner theory could bound;
%   delta_gauge is pure bookkeeping that the pivoting scrambles -- and Prop 2.4
%   says it alone is enough to destroy a frozen basis, even when M does not move
%   at all.
%
%   Cost: one dense n-by-n SVD.  Intended for n <~ 2500.
%
%   out fields:
%     .Q .C2t                    the alignment and the aligned factor
%     .delta_chart .delta_metric .delta_gauge
%     .triangle_ok               delta_chart <= delta_metric + delta_gauge + 1e-10
%     .relC   ||C2-C1||_2 / ||C1||_2          (factor motion, gauge dependent)
%     .relC_aligned ||C2-C1*Q||_2 / ||C1||_2  (the part no regauge can remove)
%     .relM   ||M2-M1||_2 / ||M1||_2          (metric motion, gauge INVARIANT)
%     .aligned_fraction  relC_aligned / relC  (SMALL => the factor motion is
%                        mostly a regauge that an alignment removes)
%
%   See also: gap, transport_V.

    C1 = full(C1);  C2 = full(C2);

    [W, ~, Z] = svd(C1' * C2);
    Q   = W * Z';                        % orthogonal, C1*Q ~= C2
    C2t = C2 * Q';                       % C2t*C2t' = M2, aligned to C1's gauge

    T  = (C2'  / C1');                   % C2' * C1^-T
    Tt = (C2t' / C1');                   % gauge-aligned transport

    V1 = orth_trunc(V1);
    W1 = Tt * V1;

    out = struct();
    out.Q            = Q;
    out.C2t          = C2t;
    out.delta_chart  = gap(V1, T * V1);
    out.delta_metric = gap(V1, W1);
    out.delta_gauge  = gap(W1, Q' * W1);
    out.triangle_ok  = out.delta_chart <= out.delta_metric + out.delta_gauge + 1e-10;

    nC1 = norm(C1, 2);
    M1  = C1 * C1';
    M2  = C2 * C2';
    out.relC         = norm(C2 - C1,     2) / nC1;
    out.relC_aligned = norm(C2 - C1 * Q, 2) / nC1;
    out.relM         = norm(M2 - M1,     2) / norm(M1, 2);
    out.aligned_fraction = out.relC_aligned / max(out.relC, realmin);
end
