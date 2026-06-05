function [P, nc] = tentative_prolongator(A, theta, maxAggSize, target_nc)
%TENTATIVE_PROLONGATOR  Tentative aggregation prolongator (plain VMB, unsmoothed).
%
%   P            = tentative_prolongator(A)
%   P            = tentative_prolongator(A, theta)
%   P            = tentative_prolongator(A, theta, maxAggSize)
%   P            = tentative_prolongator(A, theta, maxAggSize, target_nc)
%   [P, nc]      = tentative_prolongator(...)
%
%   Builds the Vaněk-Mandel-Brezina tentative prolongator P_0 from the
%   strength graph of an SPD matrix A.  No smoothing is applied; for
%   smoothed aggregation see +src/+precond/make_amg_preconditioner.m's
%   nested build_sa_prolongator with omegaInterp > 0.
%
%   Inputs
%     A          : n-by-n materialized sparse SPD matrix.  Function
%                  handles are not supported (aggregation needs entry-wise
%                  access to A).
%     theta      : (optional, default 0.05) strength-of-connection
%                  threshold.  Off-diagonal entry (i,j) is "strong" iff
%                  |A(i,j)| >= theta * max_k |A(i,k)|.  The strength
%                  graph is symmetrized.
%     maxAggSize : (optional, default 16) hard cap on aggregate size.
%                  Roots absorb up to (maxAggSize - 1) of their unassigned
%                  strong neighbors.  When target_nc is provided, this is
%                  also the hard upper bound for the bisection search.
%     target_nc  : (optional, default []) desired number of coarse
%                  columns.  When given (positive integer in (0, n]) the
%                  per-aggregate cap is bisected within [1, maxAggSize]
%                  to land within 5% of target_nc.  The strength graph
%                  is built only once, so this is much cheaper than
%                  bisecting externally.  If target_nc is unreachable
%                  within maxAggSize (i.e. target_nc < ceil(n/maxAggSize))
%                  the closest reachable nc is returned (best-effort);
%                  raise maxAggSize if you need a smaller nc.
%
%   Outputs
%     P          : sparse n-by-nc 0/1 indicator matrix.  Each row has
%                  exactly one 1 (every fine-grid node belongs to
%                  exactly one aggregate).  Column j is the indicator
%                  of aggregate j.
%     nc         : size(P, 2).  Convenience for bisection callers.
%
%   Properties
%     - nnz(P) = n.
%     - ||P(:,j)||_2 = sqrt(|aggregate j|) — columns NOT orthonormal.
%     - span(P) is the piecewise-constant subspace over aggregates;
%       captures the constant near-nullspace and the lowest-frequency
%       modes of an SPD elliptic A by construction.
%     - To orthonormalize columns:
%         P = P * spdiags(1./sqrt(full(sum(P,1)))', 0, nc, nc);
%
%   Defaults match the project convention in
%   +src/+precond/make_amg_preconditioner.m (theta=0.05, maxAggSize=16).
%
%   See also: subspace_capture, +src.precond.make_amg_preconditioner.

    if nargin < 2 || isempty(theta),      theta      = 0.05; end
    if nargin < 3 || isempty(maxAggSize), maxAggSize = 16;   end
    if nargin < 4,                        target_nc  = [];   end

    if ~ismatrix(A) || ~isnumeric(A)
        error('tentative_prolongator:notMatrix', ...
              ['tentative_prolongator requires a materialized matrix; '  ...
               'function handles are not supported because aggregation ' ...
               'needs entry-wise access to A.']);
    end
    if ~issparse(A), A = sparse(A); end

    n = size(A, 1);

    if ~isempty(target_nc)
        if ~isnumeric(target_nc) || ~isscalar(target_nc) || ...
           target_nc <= 0 || target_nc > n || target_nc ~= floor(target_nc)
            error('tentative_prolongator:badTarget', ...
                  'target_nc must be a positive integer in (0, %d]; got %s.', ...
                  n, mat2str(target_nc));
        end
    end

    S = build_strength_graph(A, theta);

    if isempty(target_nc)
        [P, nc] = aggregate_with_cap(S, n, maxAggSize);
        return;
    end

    % --- Bisect cap in [1, maxAggSize] to hit target_nc ---
    lo = 1;
    hi = maxAggSize;
    [P_lo, nc_lo] = aggregate_with_cap(S, n, lo);  % nc_lo = n (singletons)
    [P_hi, nc_hi] = aggregate_with_cap(S, n, hi);  % smallest reachable nc

    if nc_lo <= target_nc
        P = P_lo;  nc = nc_lo;  return;
    end
    if nc_hi >= target_nc
        P = P_hi;  nc = nc_hi;  return;
    end

    tol = max(0.05 * target_nc, 1);
    for it = 1:25
        mid = max(lo + 1, floor((lo + hi) / 2));
        if mid == lo || mid == hi, break; end
        [P_mid, nc_mid] = aggregate_with_cap(S, n, mid);
        if nc_mid >= target_nc
            lo = mid;  P_lo = P_mid;  nc_lo = nc_mid;
        else
            hi = mid;  P_hi = P_mid;  nc_hi = nc_mid;
        end
        if min(abs(nc_lo - target_nc), abs(nc_hi - target_nc)) <= tol
            break;
        end
    end

    if abs(nc_lo - target_nc) <= abs(nc_hi - target_nc)
        P = P_lo;  nc = nc_lo;
    else
        P = P_hi;  nc = nc_hi;
    end
end


function S = build_strength_graph(A, theta)
%BUILD_STRENGTH_GRAPH  Symmetrized 0/1 strength graph: |A_ij| >= theta * max_k |A_ik|.
    n            = size(A, 1);
    d            = diag(A);
    Aoff         = A - spdiags(d, 0, n, n);
    [ii, jj, vv] = find(Aoff);
    absvals      = abs(vv);
    rowMax       = accumarray(ii, absvals, [n, 1], @max, 0);
    strong       = absvals >= theta * rowMax(ii);

    S = sparse(ii(strong), jj(strong), 1, n, n);
    S = spones(S + S');
end


function [P, nc] = aggregate_with_cap(S, n, cap)
%AGGREGATE_WITH_CAP  Greedy degree-ordered aggregation with per-aggregate size cap.
    agg      = zeros(n, 1);
    assigned = false(n, 1);
    nc       = 0;
    deg      = full(sum(S, 2));
    [~, ord] = sort(deg, 'descend');

    for t = 1:n
        i = ord(t);
        if assigned(i), continue; end

        nc    = nc + 1;
        neigh = find(S(i, :));
        neigh = neigh(~assigned(neigh));
        nodes = [i; neigh(:)];
        nodes = nodes(1:min(numel(nodes), cap));

        agg(nodes)      = nc;
        assigned(nodes) = true;
    end

    missed = find(~assigned);
    for k = 1:numel(missed)
        nc = nc + 1;
        agg(missed(k)) = nc;
    end

    P = sparse((1:n)', agg, 1, n, nc);
end
