function [Mfun, levels] = make_amg_preconditioner(A, varargin)
%MAKE_AMG_PRECONDITIONER  Cheap-setup AMG approximate inverse.
%
% Usage:
%   [Mfun, levels] = make_amg_preconditioner(A);
%   y = Mfun(r);                    % approximately A^{-1} r
%   [x,flag] = pcg(A,b,1e-8,200,Mfun);
%
% This version prioritizes low setup/build cost over AMG quality.

    p = inputParser;

    % Cheap-build defaults
    p.addParameter('maxLevels', 3);
    p.addParameter('minCoarseSize', 800);
    p.addParameter('theta', 0.05);
    p.addParameter('omegaSmooth', 2/3);
    p.addParameter('omegaInterp', 0.0);     % 0 = no prolongation smoothing
    p.addParameter('preSmooth', 1);
    p.addParameter('postSmooth', 1);
    p.addParameter('maxAggSize', 16);
    p.addParameter('coarseSolve', 'backslash'); % lowest build cost
    p.addParameter('fineSmootherL', []);   % external ICHOL factor (lower)
    p.addParameter('fineSmootherLt', []);  % optional precomputed L'

    p.parse(varargin{:});
    opts = p.Results;

    if ~issparse(A)
        A = sparse(A);
    end

    A = 0.5 * (A + A');   % enforce symmetry numerically

    levels = build_hierarchy(A, opts);

    Mfun = @(r) vcycle(levels, 1, r, opts);
end


function levels = build_hierarchy(A, opts)

    levels = struct([]);

    % Resolve fine-level ICHOL factors once (precompute L' if not supplied).
    if ~isempty(opts.fineSmootherL)
        fineL = opts.fineSmootherL;
        if isempty(opts.fineSmootherLt)
            fineLt = fineL';
        else
            fineLt = opts.fineSmootherLt;
        end
    else
        fineL  = [];
        fineLt = [];
    end

    for ell = 1:opts.maxLevels
        n = size(A,1);

        levels(ell).A = A;
        levels(ell).Dinv = 1 ./ diag(A);
        levels(ell).P = [];
        levels(ell).R = [];
        levels(ell).coarseType = '';
        levels(ell).L = [];
        levels(ell).fineL  = [];
        levels(ell).fineLt = [];

        if ell == 1
            levels(1).fineL  = fineL;
            levels(1).fineLt = fineLt;
        end

        if n <= opts.minCoarseSize || ell == opts.maxLevels
            assign_coarse_solver(ell);
            break;
        end

        if ell == 1
            P = build_sa_prolongator(A, opts.theta, opts.omegaInterp, ...
                                     opts.maxAggSize, fineL, fineLt);
        else
            P = build_sa_prolongator(A, opts.theta, opts.omegaInterp, ...
                                     opts.maxAggSize, [], []);
        end
        nc = size(P,2);

        % Stop if coarsening is not effective.
        if nc >= 0.8 * n || nc < 1
            assign_coarse_solver(ell);
            break;
        end

        Ac = P' * (A * P);
        Ac = 0.5 * (Ac + Ac');

        levels(ell).P = P;
        levels(ell).R = P';

        A = sparse(Ac);
    end

    function assign_coarse_solver(idx)
        switch lower(opts.coarseSolve)
            case 'chol'
                try
                    levels(idx).L = chol(levels(idx).A, 'lower');
                    levels(idx).coarseType = 'chol';
                catch
                    levels(idx).coarseType = 'backslash';
                end
            otherwise
                levels(idx).coarseType = 'backslash';
        end
    end
end


function x = vcycle(levels, ell, b, opts)

    A = levels(ell).A;

    % Coarsest level.
    if ~isfield(levels(ell), 'P') || isempty(levels(ell).P)
        x = coarse_solve(levels(ell), b);
        return;
    end

    x = zeros(size(b));

    % Pre-smoothing.
    x = smooth(levels(ell), A, b, x, opts.omegaSmooth, opts.preSmooth);

    % Coarse correction.
    r = b - A * x;
    rc = levels(ell).R * r;
    ec = vcycle(levels, ell + 1, rc, opts);
    x = x + levels(ell).P * ec;

    % Post-smoothing.
    x = smooth(levels(ell), A, b, x, opts.omegaSmooth, opts.postSmooth);
end


function x = smooth(level, A, b, x, omega, nsweeps)
    if ~isempty(level.fineL)
        L  = level.fineL;
        Lt = level.fineLt;
        for k = 1:nsweeps
            r = b - A * x;
            x = x + (Lt \ (L \ r));
        end
    else
        for k = 1:nsweeps
            r = b - A * x;
            x = x + omega * (level.Dinv .* r);
        end
    end
end


function x = coarse_solve(level, b)

    switch level.coarseType
        case 'chol'
            L = level.L;
            x = L' \ (L \ b);

        otherwise
            x = level.A \ b;
    end
end


function P = build_sa_prolongator(A, theta, omegaInterp, maxAggSize, L, Lt)
%BUILD_SA_PROLONGATOR  Cheap smoothed-aggregation prolongator.
%
% If omegaInterp = 0, this returns the tentative aggregation prolongator P0.
% That minimizes build cost.
%
% If L (and Lt = L') are non-empty and omegaInterp > 0, the prolongator is
% smoothed using the ICHOL factor on the fine level:
%     P = P0 - omegaInterp * (L L')^{-1} (A * P0).
% Otherwise the standard Jacobi-style D^{-1} smoother is used.

    n = size(A,1);

    d = diag(A);
    Aoff = A - spdiags(d, 0, n, n);

    [ii, jj, vv] = find(Aoff);
    absvals = abs(vv);

    rowMax = accumarray(ii, absvals, [n,1], @max, 0);
    strong = absvals >= theta * rowMax(ii);

    S = sparse(ii(strong), jj(strong), 1, n, n);
    S = spones(S + S');

    agg = zeros(n,1);
    assigned = false(n,1);
    nc = 0;

    deg = full(sum(S,2));
    [~, order] = sort(deg, 'descend');

    for t = 1:n
        i = order(t);

        if assigned(i)
            continue;
        end

        nc = nc + 1;

        neigh = find(S(i,:));
        neigh = neigh(~assigned(neigh));

        nodes = [i; neigh(:)];
        nodes = nodes(1:min(numel(nodes), maxAggSize));

        agg(nodes) = nc;
        assigned(nodes) = true;
    end

    missed = find(~assigned);
    for k = 1:numel(missed)
        nc = nc + 1;
        agg(missed(k)) = nc;
    end

    % Tentative prolongator.
    P0 = sparse((1:n)', agg, 1, n, nc);

    % Cheapest option: no prolongation smoothing.
    if omegaInterp == 0
        P = P0;
        return;
    end

    if nargin >= 6 && ~isempty(L)
        % ICHOL-smoothed prolongator on the fine level:
        %   P = P0 - omegaInterp * (L L')^{-1} (A * P0)
        P = P0 - omegaInterp * (Lt \ (L \ (A * P0)));
    else
        % Jacobi-style smoothed aggregation.
        Dinv = 1 ./ diag(A);
        P = P0 - omegaInterp * spdiags(Dinv, 0, n, n) * (A * P0);
    end

    colNorm = sqrt(full(sum(P.^2, 1)))';
    colNorm(colNorm == 0) = 1;
    P = P * spdiags(1 ./ colNorm, 0, nc, nc);

    P = sparse(P);
end