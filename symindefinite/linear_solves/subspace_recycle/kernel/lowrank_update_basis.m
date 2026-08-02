function [What, info, ctx] = lowrank_update_basis(S, n, P_n, ctx, opts)
%LOWRANK_UPDATE_BASIS  The missing deflation component, as a rank-<=2nC block.
%
%   [What, info, ctx] = LOWRANK_UPDATE_BASIS(S, N, P_N, CTX, OPTS)
%
%   S is a build_stokes_sequence output, N the step, P_N the ILDL struct whose
%   split coordinates the block must live in, CTX a persistent context ([] on the
%   first call) carrying the frozen reference factorization.
%
%   WHY THIS BLOCK.  With the coordinates held fixed, the split operator obeys
%
%       Ahat_n = Ahat_ref + Uhat B Uhat',   Uhat = C^-1 [dC, Sel],  B = [0 I; I 0]
%
%   so exactly (not to first order): any x with Uhat'x = 0 keeps its eigenpair
%   verbatim, and every eigenvector that DOES move lies in the resolvent image
%   range((Ahat_ref - lambda I)^-1 Uhat).  The deflation targets sit nearest
%   lambda = 0, which gives the unshifted augmentation below; opts.shifts adds
%   the shifted resolvents when that approximation is not enough.
%
%   WHY IT IS CHEAP.  Substituting Ahat^-1 = C' K^-1 C, the C^-1 cancels:
%
%       Ahat_n^-1 Uhat = C' * ( K_n^-1 [dC, Sel] )
%
%   and because K_n - K_ref maps everything into range([dC, Sel]), one has the
%   exact span identity
%
%       K_n^-1 * range([dC, Sel])  ==  K_ref^-1 * range([dC, Sel]).
%
%   (Woodbury gives K_n^-1 U = Y0 * (Cap \ B) with Y0 = K_ref^-1 U; the right
%   factor is invertible, so it moves the columns but not the span.)  A deflation
%   basis only cares about the span, so a FROZEN factorization of K_ref is exact
%   here — no refactorization, and no exposure to an ill-conditioned capacitance.
%   Sel is time-independent, so K_ref^-1 Sel is computed once and cached: the
%   per-step cost is nC (<= 44) backsolves and one n-by-2nC QR.
%
%   MODES (opts.mode):
%     'invref'     (default) C_n' * (K_ref \ [dC, Sel]) — the cheap exact-span form
%     'exactsolve' C_n' * (K_n   \ [dC, Sel]) — same span, fresh factorization;
%                  the reference implementation the unit test checks 'invref' against
%     'raw'        C_n^-1 * [dC, Sel] — the residual space itself, no solve at all
%                  (2nC triangular solves); an ablation, not the correction space
%     'shifted'    'invref' plus (K_ref - sigma*M)^-1 blocks for each sigma in
%                  opts.shifts; needs the explicit M = C C' and one extra
%                  factorization per shift (cached in CTX)
%
%   OPTS: .mode, .ref (reference step, default 1), .shifts ([]), .orth (true),
%         .Cn (precomputed C_n from ildl_coordinate_map, else rebuilt).
%
%   INFO: .mode .ncols .ncols_raw .rank_drop .n_backsolves .rcond_capacitance
%         .woodbury_relerr .time .dC_nnz .dC_normF
%
%   rcond_capacitance and woodbury_relerr are reported because a Woodbury-based
%   SOLVE (as opposed to this span computation) does depend on the capacitance
%   B + U'K_ref^-1U being well conditioned, and ||dC||/||C|| ~ 1 here.  They are
%   diagnostics for that downstream use, not preconditions for this function.
%
%   On disk_static dC is exactly zero, the generator loses half its rank and the
%   block collapses to the nC columns of C'K^-1 Sel — the falsification control.
%
%   See also: seq_dCblk, seq_K, transport_V, orth_trunc, augment_recycle_V.

    t0 = tic;
    if nargin < 5 || isempty(opts), opts = struct(); end
    mode   = getdef(opts, 'mode',   'invref');
    ref    = getdef(opts, 'ref',    1);
    shifts = getdef(opts, 'shifts', []);
    doOrth = getdef(opts, 'orth',   true);

    ctx = ensure_ctx(S, ctx, ref);
    if nargin >= 5 && isfield(opts, 'Cn') && ~isempty(opts.Cn)
        Cn = opts.Cn;
    else
        Cn = ildl_coordinate_map(P_n);
    end

    [U, dC] = seq_dCblk(S, n, ctx.ref);
    nbs = 0;

    switch lower(mode)
        case 'raw'
            % The residual space Uhat = C^-1 U itself: no solve, only triangular
            % applies.  Kept as the ablation that isolates "does the correction
            % need the inverse at all?".
            Y = P_n.applyCinv(full(U));

        case 'invref'
            [Y0, nbs] = ref_solve(ctx, dC);
            Y = Cn' * Y0;

        case 'exactsolve'
            dKn = decomposition(seq_K(S, n));
            Y   = Cn' * (dKn \ full(U));
            nbs = size(U, 2);

        case 'shifted'
            [Y0, nbs] = ref_solve(ctx, dC);
            Y = Cn' * Y0;
            for sg = shifts(:)'
                [ctx, dKs] = shifted_dec(ctx, S, sg);
                Y   = [Y, Cn' * (dKs \ full(U))];   %#ok<AGROW>
                nbs = nbs + size(U, 2);
            end

        otherwise
            error('lowrank_update_basis:unknownMode', ...
                  'unknown opts.mode "%s" (raw|invref|exactsolve|shifted)', mode);
    end

    ncols_raw = size(Y, 2);
    if doOrth
        What = orth_trunc(Y);
    else
        What = real(full(Y));
    end

    info = struct();
    info.mode              = lower(mode);
    info.ref               = ctx.ref;
    info.ncols             = size(What, 2);
    info.ncols_raw         = ncols_raw;
    info.rank_drop         = ncols_raw - size(What, 2);
    info.n_backsolves      = nbs;
    info.dC_nnz            = nnz(dC);
    info.dC_normF          = norm(dC, 'fro');
    [info.rcond_capacitance, info.woodbury_relerr] = capacitance_health(ctx, U);
    info.time              = toc(t0);
end

%==========================================================================
%  Reference context: the frozen factorization and the cached K_ref^-1 Sel
%==========================================================================
function ctx = ensure_ctx(S, ctx, ref)
    if ~isempty(ctx) && isfield(ctx, 'ref') && ctx.ref == ref
        return;
    end
    ctx        = struct();
    ctx.ref    = ref;
    ctx.dKref  = decomposition(seq_K(S, ref));
    ctx.YSel   = ctx.dKref \ full(S.Sel);     % time-independent: solved once
    ctx.Sel    = S.Sel;
    ctx.shiftK = {};                          % {sigma, decomposition} pairs
    ctx.n_backsolves_total = size(S.Sel, 2);
end

function [Y0, nbs] = ref_solve(ctx, dC)
%REF_SOLVE  K_ref^-1 * [dC, Sel] reusing the cached Sel half.
    Y0  = [ctx.dKref \ full(dC), ctx.YSel];
    nbs = size(dC, 2);                        % the Sel half costs nothing again
end

function [ctx, dKs] = shifted_dec(ctx, S, sigma)
%SHIFTED_DEC  Cached decomposition(K_ref - sigma*M_ref), M_ref = C_ref C_ref'.
    for i = 1:numel(ctx.shiftK)
        if ctx.shiftK{i}.sigma == sigma
            dKs = ctx.shiftK{i}.dec;
            return;
        end
    end
    if ~isfield(ctx, 'Mref') || isempty(ctx.Mref)
        Pref     = src.precond.make_ildl_precond(seq_K(S, ctx.ref), ...
                                                 struct('mode', 'nofill'));
        Cref     = ildl_coordinate_map(Pref);
        M        = Cref * Cref';
        ctx.Mref = (M + M') / 2;
    end
    dKs = decomposition(seq_K(S, ctx.ref) - sigma * ctx.Mref);
    ctx.shiftK{end+1} = struct('sigma', sigma, 'dec', dKs);
end

%==========================================================================
function [rc, relerr] = capacitance_health(ctx, U)
%CAPACITANCE_HEALTH  Condition of the Woodbury capacitance Cap = B + U'K_ref^-1U,
% and the residual of the identity K_n^-1 U = (K_ref^-1 U) * (Cap \ B).  Reported
% for the downstream Woodbury SOLVE; the span computation above does not use it.
    m = size(U, 2) / 2;
    if m < 1 || mod(size(U, 2), 2) ~= 0
        rc = NaN;  relerr = NaN;  return;
    end
    Bm  = [sparse(m, m), speye(m); speye(m), sparse(m, m)];
    Y0  = ctx.dKref \ full(U);
    Cap = full(Bm) + full(U' * Y0);
    rc  = rcond(Cap);
    if ~isfinite(rc) || rc < eps
        relerr = NaN;
    else
        relerr = norm(Cap * (Cap \ full(Bm)) - full(Bm), 'fro') / ...
                 max(norm(full(Bm), 'fro'), eps);
    end
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
