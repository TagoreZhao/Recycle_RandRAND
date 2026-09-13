function [W, out] = varvisc_build_projected_arnoldi(Afun, V, r0, m, opts)
%VARVISC_BUILD_PROJECTED_ARNOLDI Krylov basis of (I-VV') A (I-VV').
% V is orthonormal in the SAME coordinates as Afun and r0. Two passes of
% block Gram-Schmidt against [V,W] implement augmented Arnoldi without a
% dense projector. AW = V*B + W*H + tail*e_last' is retained for inspection.
% status: 0=budget reached, 1=zero projected start, 2=happy/numerical
% breakdown, 3=dimension cap. No random replacement or padding is performed.
    if nargin < 5, opts = struct(); end
    if ~isfield(opts, 'breakdown_tol'), opts.breakdown_tol = 100 * eps; end
    validateattributes(m, {'numeric'}, {'scalar','integer','nonnegative','finite'});
    n = numel(r0);
    if isempty(V), V = zeros(n, 0); end
    assert(size(V, 1) == n && size(r0, 2) == 1, 'Incompatible basis/start sizes.');
    k = size(V, 2);
    budget = min(m, max(0, n-k));
    W = zeros(n, budget);
    AW = zeros(n, budget);
    H = zeros(budget, budget);
    B = zeros(k, budget);
    tail = zeros(n, 1);
    r = r0;
    for pass = 1:2, r = r - V * (V' * r); end
    beta = norm(r);
    out = struct('requested', m, 'actual', 0, 'status', 0, ...
        'start_norm', beta, 'operator_columns', 0, 'H', H, 'B', B, ...
        'AW', AW, 'tail', tail);
    if budget == 0 || beta <= opts.breakdown_tol * max(norm(r0), realmin)
        if budget < m, out.status = 3; end
        if budget > 0, out.status = 1; end
        W = zeros(n, 0); out.H = zeros(0); out.B = zeros(k,0);
        out.AW = zeros(n,0);
        return;
    end
    W(:,1) = r / beta;
    for j = 1:budget
        a = Afun(W(:,j));
        AW(:,j) = a;
        z = a;
        for pass = 1:2
            c = V' * z; B(:,j) = B(:,j) + c; z = z - V*c;
            h = W(:,1:j)' * z;
            H(1:j,j) = H(1:j,j) + h;
            z = z - W(:,1:j)*h;
        end
        hnext = norm(z);
        tail = z;
        if hnext <= opts.breakdown_tol * max(norm(a), realmin)
            out.status = 2;
            break;
        end
        if j < budget
            H(j+1,j) = hnext;
            W(:,j+1) = z / hnext;
        end
    end
    if out.status == 0 && budget < m, out.status = 3; end
    W = W(:,1:j);
    out.actual = j; out.operator_columns = j;
    out.H = H(1:j,1:j); out.B = B(:,1:j);
    out.AW = AW(:,1:j); out.tail = tail;
end
