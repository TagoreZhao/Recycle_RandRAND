function [V, info, Y] = build_lowrank_sketch_V(ctx, K, P, opts)
%BUILD_LOWRANK_SKETCH_V  Coarse space from a randomized sketch of D = A_1^{-1}(A_2 - A_1).
%
%   [V, info, Y] = BUILD_LOWRANK_SKETCH_V(CTX, K, P, OPTS)
%
%   CTX is a frozen_ldl_context for the REFERENCE system A_1 = K_ref; K is the
%   CURRENT system A_2; P is the current step's make_ildl_precond struct, whose
%   split coordinates V must live in.  Returns an orthonormal V in HAT
%   coordinates, ready for src.precond.two_level_split_solve / deflation_Psqrt_apply.
%
%   THE METHOD.  A_2 = A_1 + B, and the directions the update moves are the range
%   of D = A_1^{-1} B.  D is NOT symmetric, so its dominant invariant directions
%   are its leading LEFT SINGULAR vectors, and the standard randomized power
%   iteration for those is
%
%       Y = (D D')^q D Omega,    Omega = randn(n, k),    V = orth(C_n' Y).
%
%   D is never formed.  It is applied as a handle -- one sparse dK matvec and one
%   batched backsolve against the frozen factor per block -- so the per-step cost
%   is (2q+1)*k backsolves, (2q+1)*k dK matvecs and one n-by-k pivoted QR, with
%   NO refactorization after the reference step.
%
%   WHY THE SPAN IS KNOWN IN ADVANCE.  In this benchmark K_n - K_ref = U B U' with
%   U = [dC, Sel] and B invertible (see seq_dCblk), so
%
%       range(D) = K_ref^{-1} range(U),   dim <= 2*nC,
%
%   an exactly known subspace of exactly known dimension.  Two consequences worth
%   reporting rather than hiding: (i) k above that rank buys nothing -- orth_trunc
%   returns fewer than k columns and info.rank_drop says so, and the EFFECTIVE
%   dimension info.ncols, not k, is the number to quote in any comparison; (ii) at
%   k >= 2*nC this coarse space contains every direction the operator update can
%   have moved, which is the same span lowrank_update_basis computes in one shot --
%   this builder differs in that it TRUNCATES to the dominant k of it.
%
%   REORTHOGONALIZATION IS ON BY DEFAULT, unlike src.precond.subspace_iter_plain,
%   which deliberately skips it.  There the operator is a well-scaled inverse and
%   q is small; here D is rank-deficient by construction and typically severely
%   graded, so an unorthogonalized block loses its trailing directions within a
%   round or two and the sketch silently returns a lower-dimensional space than it
%   reports.  Turn it off (opts.reorth = false) only to demonstrate that.
%
%   COORDINATES.  Y is built in PHYSICAL coordinates and mapped into the current
%   step's hat coordinates by transport_V (= orth_trunc(C_n' Y)): MINRES runs on
%   Ahat_n = C_n^{-1} K_n C_n^{-T} with yhat = C_n^T x, so C_n^T is the map a
%   physical subspace goes through to be deflated at this step.  That map also
%   supplies the V'V = I that deflation_Psqrt_apply's (I - VV') projector needs.
%
%   OPTS: .k (sketch width, required), .q (power rounds, default 2),
%         .reorth (default true), .Cn (precomputed C_n, else rebuilt).
%
%   INFO: .k .q .ncols .ncols_raw .rank_drop .n_backsolves .n_dK_matvecs
%         .dK_nnz .dK_normF .time
%
%   dK EXACTLY ZERO (the disk_static control: the coupling never moves) returns an
%   empty V and zero cost, and the caller degrades to plain ILDL.  That is the
%   falsification case, not an error.
%
%   Kept LOCAL to this benchmark rather than promoted to
%   src.precond.build_deflation_V: that builder's (A, P, opts, dA) signature has no
%   room for a frozen REFERENCE system, which is the entire input here.
%
%   See also: frozen_ldl_context, frozen_ldl_apply, transport_V, orth_trunc,
%             src.precond.two_level_split_solve.

    t0 = tic;
    if nargin < 4 || isempty(opts), opts = struct(); end
    k      = opts.k;
    q      = getdef(opts, 'q',      2);
    reorth = getdef(opts, 'reorth', true);

    n  = size(K, 1);
    dK = K - ctx.Kref;                       % A_2 - A_1

    info            = struct();
    info.k          = k;
    info.q          = q;
    info.dK_nnz     = nnz(dK);
    info.dK_normF   = norm(dK, 'fro');
    info.ncols_raw  = k;

    if info.dK_nnz == 0 || k < 1
        % Nothing moved: D is exactly zero, there is no space to build, and an
        % empty coarse space is the honest answer (two_level_split_solve then runs
        % the plain ILDL split solve).
        V              = zeros(n, 0);
        Y              = zeros(n, 0);
        info.ncols     = 0;
        info.rank_drop = k;
        info.n_backsolves  = 0;
        info.n_dK_matvecs  = 0;
        info.time      = toc(t0);
        return;
    end

    Dfun  = @(X) frozen_ldl_apply(ctx, dK * X);          % D  = K_ref^{-1} dK
    Dtfun = @(X) dK' * frozen_ldl_apply(ctx, X);         % D' = dK' K_ref^{-1}
                                                         % (K_ref symmetric)
    nbs = 0;                                             % backsolves
    nmv = 0;                                             % sparse dK matvecs

    Y = randn(n, k);
    [Y, nbs, nmv] = step_D(Dfun, Y, nbs, nmv);
    Y = maybe_orth(Y, reorth);
    for i = 1:q
        [Z, nbs, nmv] = step_D(Dtfun, Y, nbs, nmv);
        Z = maybe_orth(Z, reorth);
        [Y, nbs, nmv] = step_D(Dfun, Z, nbs, nmv);
        Y = maybe_orth(Y, reorth);
    end
    if ~reorth
        Y = orth_trunc(Y);                               % V'V = I is not optional
    end

    Cn = getdef(opts, 'Cn', []);
    V  = transport_V(Y, P, Cn);                          % orth(C_n' Y): -> hat coords

    info.ncols        = size(V, 2);
    info.rank_drop    = k - info.ncols;
    info.n_backsolves = nbs;
    info.n_dK_matvecs = nmv;
    info.time         = toc(t0);
end

%==========================================================================
function [Y, nbs, nmv] = step_D(fun, X, nbs, nmv)
%STEP_D  One block application of D or D', with the operation count it costs.
    Y   = fun(X);
    nbs = nbs + size(X, 2);
    nmv = nmv + size(X, 2);
end

%==========================================================================
function Y = maybe_orth(Y, reorth)
    if reorth, Y = orth_trunc(Y); end
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
