function [g, info] = gap(X, Y)
%GAP  Euclidean gap between two subspaces, given ANY bases of them.
%
%   [g, info] = GAP(X, Y)
%
%   Returns the gap metric
%
%       g = || Pi_X - Pi_Y ||_2 = sin(theta_max)      (equal dimensions)
%
%   where Pi_X, Pi_Y are the ORTHOGONAL PROJECTORS onto span(X), span(Y).  The
%   value depends on X and Y only through their spans: X -> X*R with R
%   invertible leaves g unchanged.  That invariance is the whole point -- the
%   deflation method is a function on the Grassmannian (P^{1/2}(V*Q) = P^{1/2}(V)
%   for orthogonal Q), so every quantity this study reports must be one too.
%
%   Computed WITHOUT forming the n-by-n projectors, via the standard identity
%
%       || Pi_X - Pi_Y ||_2 = max( ||(I-Pi_Y) Q_X||_2 , ||(I-Pi_X) Q_Y||_2 ),
%
%   which costs two n-by-k products.  For equal dimensions the two directed
%   quantities coincide in exact arithmetic; the max is kept because they differ
%   at roundoff and because unequal dimensions are allowed (there the gap is 1
%   whenever dim X ~= dim Y, and the directed values are the informative ones).
%
%   The 2-norm gap saturates: ONE lost direction out of k already gives 1.  So
%   info.dF reports the Frobenius companion
%
%       d_F = ||Pi_X - Pi_Y||_F / sqrt(2k) = sqrt( mean_i sin^2 theta_i ),
%
%   the root-mean-square principal angle, also a function of the projectors
%   alone and therefore equally basis-free.  Both are reported everywhere in
%   this study: d says whether ANY direction was lost, d_F says how much of the
%   space was.
%
%   info fields:
%     .dX     ||(I-Pi_Y) Q_X||_2, the directed distance from span(X) to span(Y)
%     .dY     ||(I-Pi_X) Q_Y||_2, the reverse
%     .dF     the Frobenius-normalized gap defined above
%     .angles principal angles (radians), ascending
%     .kX .kY numerical ranks actually used
%
%   See also: gap_M, orth_trunc, subspace_capture_directed.

    QX = orth_trunc(X);
    QY = orth_trunc(Y);

    info = struct('dX', NaN, 'dY', NaN, 'dF', NaN, 'angles', [], ...
                  'kX', size(QX, 2), 'kY', size(QY, 2));

    if info.kX == 0 || info.kY == 0
        g = double(info.kX ~= info.kY);
        return;
    end

    G = QY' * QX;                       % kY-by-kX, the cosine block
    info.dX = norm(QX - QY * G, 2);
    info.dY = norm(QY - QX * G', 2);

    s = svd(G);
    s = min(max(s, 0), 1);
    info.angles = acos(s(end:-1:1));    % ascending principal angles
    info.dF     = sqrt(mean(sin(info.angles).^2));

    g = max(info.dX, info.dY);
    g = min(g, 1);                      % the gap never exceeds 1
end
