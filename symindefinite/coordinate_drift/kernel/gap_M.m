function [g, info] = gap_M(X, Y, M)
%GAP_M  The M-gap between two subspaces, for an SPD weight M.
%
%   [g, info] = GAP_M(X, Y, M)
%
%   Same object as GAP but with orthogonality measured in the M-inner product
%   <u,v>_M = u'*M*v:
%
%       g = || Pi^M_X - Pi^M_Y ||_M ,     Pi^M_X = X (X'MX)^{-1} X' M .
%
%   Deliberately implemented from M ALONE -- M-orthonormalize each basis with a
%   Cholesky of the k-by-k Gram matrix, then measure the directed residual in
%   the M-norm.  It never touches a factor C with M = C*C'.  That independence
%   is what makes the isometry check in exp1 a real test rather than a tautology:
%   GAP_M(X,Y,M) and GAP(C'X, C'Y) are computed by disjoint code paths and must
%   still agree to roundoff.
%
%   For a matrix Z, ||Z||_M denotes the induced norm sqrt(lambda_max(Z'MZ)), so
%   every norm here reduces to a k-by-k symmetric eigenvalue problem.
%
%   info fields:  .dX .dY .dF .angles .kX .kY   (as in GAP; .dF is the
%   Frobenius-normalized companion sqrt(mean sin^2 theta_i^M), reported because
%   the 2-norm gap saturates at 1 as soon as a single direction is lost).
%
%   See also: gap, orth_trunc.

    QX = m_orth(X, M);
    QY = m_orth(Y, M);

    info = struct('dX', NaN, 'dY', NaN, 'dF', NaN, 'angles', [], ...
                  'kX', size(QX, 2), 'kY', size(QY, 2));
    if info.kX == 0 || info.kY == 0
        g = double(info.kX ~= info.kY);
        return;
    end

    G = QY' * (M * QX);                 % kY-by-kX, M-cosine block

    info.dX = m_norm(QX - QY * G,  M);
    info.dY = m_norm(QY - QX * G', M);

    s = min(max(svd(G), 0), 1);
    info.angles = acos(s(end:-1:1));
    info.dF     = sqrt(mean(sin(info.angles).^2));

    g = min(max(info.dX, info.dY), 1);
end

%==========================================================================
function Q = m_orth(X, M)
%M_ORTH  M-orthonormal basis of span(X): Q'MQ = I, span(Q) = span(X).
%   Rank-truncates first (in the Euclidean sense, which does not change the
%   span) so a rank-deficient input cannot produce a singular Gram matrix.
    Q = orth_trunc(X);
    if isempty(Q), return; end
    G = Q' * (M * Q);
    G = (G + G') / 2;
    [U, D] = eig(full(G));
    d   = real(diag(D));
    tol = max(size(G)) * eps(max(d));
    k   = sum(d > tol);
    if k < numel(d)                     % defensive: M-singular directions
        [~, ord] = sort(d, 'descend');
        U = U(:, ord(1:k));  d = d(ord(1:k));
    end
    Q = Q * (U .* (1 ./ sqrt(d(:)))');
end

function v = m_norm(Z, M)
%M_NORM  Induced norm sqrt(lambda_max(Z'MZ)) of an n-by-k block.
    if isempty(Z), v = 0; return; end
    G = Z' * (M * Z);
    G = (G + G') / 2;
    v = sqrt(max(0, max(real(eig(full(G))))));
end
