function [Mfun, info] = make_amg_prec_ablate(A, varargin)
%MAKE_AMG_PREC_ABLATE  AMG V-cycle approximate inverse with ablatable knobs.
%
%   [Mfun, info] = make_amg_prec_ablate(A, ...) builds a smoothed-aggregation
%   AMG V-cycle preconditioner Mfun = @(r) ~ A^{-1} r, extending
%   src.precond.make_amg_preconditioner with INEXACT coarse ("inner") solves
%   and cost/diagnostic reporting, for the amg_subspace capture ablation.
%
%   LOCAL trial version; once validated the extra options may be promoted to
%   +src/+precond/make_amg_preconditioner.m.
%
%   Options identical to the +src factory:
%     maxLevels (3), minCoarseSize (800), theta (0.05), omegaSmooth (2/3),
%     omegaInterp (0 = unsmoothed tentative prolongator), preSmooth (1),
%     postSmooth (1), maxAggSize (16), fineSmootherL/fineSmootherLt
%     (external ICHOL factor used as the level-1 smoother; coarser levels
%     always smooth with damped Jacobi).
%
%   Extra options (this variant only):
%     coarseSolve        'chol' (default) | 'backslash'  exact coarsest solve
%                        'jacobi'  nu damped-Jacobi sweeps from x = 0
%                        'pcg'     loose-tolerance PCG, column by column
%     coarseJacobiSweeps nu for 'jacobi' (default 2)
%     coarsePcgTol       tolerance for 'pcg' (default 1e-2)
%     coarsePcgMaxit     iteration cap for 'pcg' (default 50)
%     projector          'sa' (default) | 'sjlt'.  'sjlt' replaces the LEVEL-1
%                        smoothed-aggregation tentative prolongator with an
%                        SJLT sketch Omega (src.precond.sjlt): each fine row
%                        gets sjltNnzPerCol entries +-1/sqrt(s) at random
%                        coarse columns -- the random, connectivity-blind
%                        counterpart of the SA tentative's 1-per-row 0/1
%                        structure.  Unlike SA, its coarse size is a FREE
%                        parameter.  Deeper levels (if any) still use SA.
%                        The sketch is drawn from the CURRENT rng state --
%                        seed rng before calling for reproducibility.
%     sjltNc             coarse size nc for 'sjlt', 1 <= nc <= n (required)
%     sjltNnzPerCol      s = nnz per fine row for 'sjlt' (default 4);
%                        s = 1 is a CountSketch = random aggregation
%
%   Second output info (replaces the raw `levels` of the +src factory):
%     .levels        per-level struct array: n, nnzA, nnzP (0 on coarsest)
%     .nLevels       number of levels actually built
%     .coarseN       coarsest-level size
%     .coarseNnz     nnz of the coarsest operator
%     .coarseType    resolved coarse solver ('chol'|'backslash'|'jacobi'|'pcg')
%     .setupTime     wall time of hierarchy build + coarse factorization [s]
%     .workPerApply  flop proxy for ONE single-vector V-cycle apply
%     .workUnits     workPerApply / nnz(A)  (1 WU = one fine matvec)
%     .opts          resolved options
%
%   Caveats the caller must know:
%     * preSmooth ~= postSmooth makes the V-cycle operator M NONSYMMETRIC
%       (valid for subspace iteration, NOT as a pcg preconditioner).
%     * preSmooth == postSmooth == 0 is pure coarse-grid correction:
%       rank(M) = coarseN, so any captured subspace lives in an
%       coarseN-dimensional range.
%
%   Mfun accepts blocks: Mfun(R) with R n-by-m applies the V-cycle to every
%   column at once (all internal ops are matrix-valued).

    p = inputParser;

    p.addParameter('maxLevels', 3);
    p.addParameter('minCoarseSize', 800);
    p.addParameter('theta', 0.05);
    p.addParameter('omegaSmooth', 2/3);
    p.addParameter('omegaInterp', 0.0);     % 0 = no prolongation smoothing
    p.addParameter('preSmooth', 1);
    p.addParameter('postSmooth', 1);
    p.addParameter('maxAggSize', 16);
    p.addParameter('coarseSolve', 'chol');
    p.addParameter('coarseJacobiSweeps', 2);
    p.addParameter('coarsePcgTol', 1e-2);
    p.addParameter('coarsePcgMaxit', 50);
    p.addParameter('projector', 'sa');
    p.addParameter('sjltNc', []);
    p.addParameter('sjltNnzPerCol', 4);
    p.addParameter('fineSmootherL', []);   % external ICHOL factor (lower)
    p.addParameter('fineSmootherLt', []);  % optional precomputed L'

    p.parse(varargin{:});
    opts = p.Results;

    if strcmpi(opts.projector, 'sjlt')
        nc = opts.sjltNc;
        if isempty(nc) || ~isscalar(nc) || nc < 1 || nc > size(A,1) ...
                || nc ~= round(nc)
            error('make_amg_prec_ablate:badSjltNc', ...
                  'projector=''sjlt'' requires integer sjltNc in [1, n].');
        end
    elseif ~strcmpi(opts.projector, 'sa')
        error('make_amg_prec_ablate:badProjector', ...
              'projector must be ''sa'' or ''sjlt'', got ''%s''.', ...
              opts.projector);
    end

    if ~issparse(A)
        A = sparse(A);
    end

    A = 0.5 * (A + A');   % enforce symmetry numerically

    t0 = tic;
    levels = build_hierarchy(A, opts);
    setupTime = toc(t0);

    Mfun = @(r) vcycle(levels, 1, r, opts);
    info = build_info(levels, opts, setupTime, nnz(A));
end


function levels = build_hierarchy(A, opts)
% Identical to src.precond.make_amg_preconditioner except for the extended
% assign_coarse_solver (jacobi / pcg branches keep the coarse operator only).

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

        if ell == 1 && strcmpi(opts.projector, 'sjlt')
            P = build_sjlt_projector(n, opts.sjltNc, opts.sjltNnzPerCol);
        elseif ell == 1
            P = build_sa_prolongator(A, opts.theta, opts.omegaInterp, ...
                                     opts.maxAggSize, fineL, fineLt);
        else
            P = build_sa_prolongator(A, opts.theta, opts.omegaInterp, ...
                                     opts.maxAggSize, [], []);
        end
        nc = size(P,2);

        % Stop if coarsening is not effective.  An sjlt level-1 size is a
        % deliberate choice (may legitimately approach n), so the
        % effectiveness heuristic only applies to SA levels.
        isSjltLevel = (ell == 1 && strcmpi(opts.projector, 'sjlt'));
        if ~isSjltLevel && (nc >= 0.8 * n || nc < 1)
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
            case 'jacobi'
                levels(idx).coarseType = 'jacobi';
            case 'pcg'
                levels(idx).coarseType = 'pcg';
            otherwise
                levels(idx).coarseType = 'backslash';
        end
    end
end


function x = vcycle(levels, ell, b, opts)

    A = levels(ell).A;

    % Coarsest level.
    if ~isfield(levels(ell), 'P') || isempty(levels(ell).P)
        x = coarse_solve(levels(ell), b, opts);
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


function x = coarse_solve(level, b, opts)

    switch level.coarseType
        case 'chol'
            L = level.L;
            x = L' \ (L \ b);

        case 'jacobi'
            % nu damped-Jacobi sweeps from x = 0 (block-capable).
            x = zeros(size(b));
            for k = 1:opts.coarseJacobiSweeps
                r = b - level.A * x;
                x = x + opts.omegaSmooth * (level.Dinv .* r);
            end

        case 'pcg'
            % Loose-tolerance PCG; MATLAB pcg is single-RHS, loop columns.
            x = zeros(size(b));
            for j = 1:size(b, 2)
                [xj, ~] = pcg(level.A, b(:, j), ...
                              opts.coarsePcgTol, opts.coarsePcgMaxit);
                x(:, j) = xj;
            end

        otherwise
            x = level.A \ b;
    end
end


function info = build_info(levels, opts, setupTime, nnzFine)
%BUILD_INFO  Per-level diagnostics + a flop proxy for one V-cycle apply.
%
% Work model per single-vector apply, per non-coarsest level ell:
%   smoothing:  (pre+post) * (nnz(A_ell) + solveCost_ell)
%               solveCost = 2*nnz(L_ichol) for the ICHOL level (two
%               triangular solves), n_ell for damped-Jacobi levels;
%   coarse correction residual: nnz(A_ell);
%   transfers: 2*nnz(P_ell) (restrict + prolong).
% Coarsest level:
%   chol      : 2*nnz(cholFactor)      (two triangular solves)
%   backslash : 2*nnz(A_c) as a crude direct-solve proxy
%   jacobi    : nu * (nnz(A_c) + n_c)
%   pcg       : maxit * (nnz(A_c) + 5*n_c)  upper-bound proxy

    nL = numel(levels);
    lv = repmat(struct('n', 0, 'nnzA', 0, 'nnzP', 0), 1, nL);
    work = 0;

    for ell = 1:nL
        lv(ell).n    = size(levels(ell).A, 1);
        lv(ell).nnzA = nnz(levels(ell).A);
        if ~isempty(levels(ell).P)
            lv(ell).nnzP = nnz(levels(ell).P);
        end

        if ell < nL
            if ~isempty(levels(ell).fineL)
                solveCost = 2 * nnz(levels(ell).fineL);
            else
                solveCost = lv(ell).n;
            end
            nSweeps = opts.preSmooth + opts.postSmooth;
            work = work + nSweeps * (lv(ell).nnzA + solveCost) ...
                        + lv(ell).nnzA + 2 * lv(ell).nnzP;
        else
            switch levels(ell).coarseType
                case 'chol'
                    work = work + 2 * nnz(levels(ell).L);
                case 'jacobi'
                    work = work + opts.coarseJacobiSweeps ...
                                * (lv(ell).nnzA + lv(ell).n);
                case 'pcg'
                    work = work + opts.coarsePcgMaxit ...
                                * (lv(ell).nnzA + 5 * lv(ell).n);
                otherwise
                    work = work + 2 * lv(ell).nnzA;
            end
        end
    end

    info = struct();
    info.projector    = opts.projector;
    info.sjltNnzPerCol = opts.sjltNnzPerCol;
    info.levels       = lv;
    info.nLevels      = nL;
    info.coarseN      = lv(nL).n;
    info.coarseNnz    = lv(nL).nnzA;
    info.coarseType   = levels(nL).coarseType;
    info.setupTime    = setupTime;
    info.workPerApply = work;
    info.workUnits    = work / nnzFine;
    info.opts         = opts;
end


function P = build_sjlt_projector(n, nc, s)
%BUILD_SJLT_PROJECTOR  n-by-nc SJLT sketch usable as a coarse projection.
%   Wraps src.precond.sjlt (transpose path: each fine row gets s entries
%   +-1/sqrt(s) at random coarse columns) and repairs two defects that
%   would make the Galerkin operator Omega'*A*Omega singular:
%     * sparse() infers dimensions from the max drawn index, so trailing
%       coarse columns that were never drawn silently shrink the matrix --
%       pad back to n-by-nc;
%     * a coarse column can end up all-zero (P(untouched) ~ exp(-n*s/nc),
%       non-negligible once nc is a sizable fraction of n) -- give each
%       empty column one +-1/sqrt(s) entry at a distinct random row.
%   Draws from the CURRENT rng state; callers seed rng for reproducibility.

    P = src.precond.sjlt(n, nc, s);
    if size(P, 1) < n || size(P, 2) < nc
        P(n, nc) = 0;
    end

    % Repair empty columns by STEALING an entry from a column that has >= 2
    % (rather than adding a new entry: an added single-entry column +-e_r can
    % duplicate an existing singleton column -- for s = 1 that makes Omega
    % rank-deficient, whereas stealing preserves the s-per-row partition
    % structure, so column supports stay disjoint and Omega keeps full rank).
    % Donor supply is guaranteed: nnz(P) = n*s >= nc >= (#nonempty + #empty).
    P = fix_empty(P, 'cols');
    % Same repair for empty ROWS: they only occur on the nc >= n direct
    % construction path (s nnz per COLUMN there; the transpose path used for
    % nc < n gives every fine row exactly s entries by construction).  An
    % empty row makes a square Omega singular.  Row repair moves entries
    % between rows within their column, so it cannot re-create empty columns.
    P = fix_empty(P, 'rows');
end


function P = fix_empty(P, which)
%FIX_EMPTY  Move one existing entry into each empty column (or row) of P.
    if strcmp(which, 'rows')
        P = fix_empty(P', 'cols')';
        return;
    end
    [n, nc] = size(P);
    empty = find(full(sum(P ~= 0, 1)) == 0);
    if isempty(empty)
        return;
    end
    [ri, ci, vi] = find(P);
    [ci, order] = sort(ci);
    ri = ri(order);  vi = vi(order);
    isFirst  = [true; diff(ci) ~= 0];
    donorIdx = find(~isFirst);          % every column keeps its first entry
    if numel(donorIdx) < numel(empty)
        error('make_amg_prec_ablate:sjltRepairFailed', ...
              'cannot repair %d empty cols with %d donor entries.', ...
              numel(empty), numel(donorIdx));
    end
    take = donorIdx(1:numel(empty));
    ci(take) = empty(:);
    P = sparse(ri, ci, vi, n, nc);
end


function P = build_sa_prolongator(A, theta, omegaInterp, maxAggSize, L, Lt)
%BUILD_SA_PROLONGATOR  Cheap smoothed-aggregation prolongator.
%   Copied verbatim from src.precond.make_amg_preconditioner.
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
