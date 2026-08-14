function [V, info, Y] = varvisc_build_Esketch_V(ctx, K, P, opts)
%VARVISC_BUILD_ESKETCH_V  Coarse space from a randomized eigen-sketch of the
% symmetric update operator E = C_ref^{-1} (A_2 - A_1) C_ref^{-T}.
%
%   [V, info, Y] = VARVISC_BUILD_ESKETCH_V(CTX, K, P, OPTS)
%
%   CTX is a varvisc_esketch_ref_context for the REFERENCE system A_1 = K_ref;
%   K is the CURRENT system A_2; P is the current step's make_ildl_precond
%   struct, whose split coordinates V must live in.  Returns an orthonormal V
%   in the CURRENT step's HAT coordinates, ready for
%   src.precond.two_level_split_solve / deflation_Psqrt_apply.
%
%   THE METHOD.  MINRES runs on the split operator Ahat = C^{-1} A_2 C^{-T},
%   and with the exact reference factor (C_ref C_ref' = |A_1|) the operator the
%   update actually perturbs it by is
%
%       E = C_ref^{-1} (A_2 - A_1) C_ref^{-T},    Ahat_ref = sign(D) + E,
%
%   which is SYMMETRIC: its dominant invariant directions are EIGENVECTORS, and
%   the plain randomized power iteration for those is
%
%       Y = E^(2q+1) Omega,    Omega = randn(n, k),    reorth between applies.
%
%   E is never formed.  It is applied as a handle -- one C_ref^{-T} triangular
%   backsolve, one sparse dK matvec and one C_ref^{-1} triangular backsolve per
%   block -- so the per-step cost is (2q+1)*k E-applies with NO refactorization
%   after the reference step; the apply count matches the predecessor D-sketch's
%   (D D')^q D exactly, keeping the arm parameter-identical to the gaussian /
%   sjlt deflation sketches (k = SKETCH_OVERSAMPLE * DEFLAT_SM_EIG, q = DEFLAT_Q).
%
%   WHY E AND NOT D = A_1^{-1} dK (the predecessor).  D = C_ref^{-T} E C_ref^T
%   is SIMILAR to E, so eigenspaces map through C^T exactly -- but D is
%   nonsymmetric and the old sketch took its leading LEFT SINGULAR vectors,
%   and singular spaces are NOT similarity-equivariant: orth(C_n^T *
%   left-singular-space(D)) is the dominant eigenspace of E only when C_ref is
%   near-orthogonal, which a graded LDL factor of a 100:1-contrast KKT matrix
%   is not.  Sketching E directly removes that metric mismatch: the sketch,
%   the deflation projector and MINRES all live in the same hat geometry.
%
%   COORDINATES / TRANSPORT.  Y is an (approximate) eigenbasis of E in the
%   REFERENCE factor's hat coordinates.  The physical subspace it denotes is
%   U = C_ref^{-T} Y, and the current step's smoother C_n re-expresses it as
%   V = orth(C_n^T U) (transport_V) -- the similarity transport
%   C_n^T C_ref^{-T}, which preserves the physical span exactly and supplies
%   the V'V = I that deflation_Psqrt_apply's (I - VV') projector needs.
%
%   REORTHOGONALIZATION IS ON BY DEFAULT: E is rank-deficient by construction
%   (rank <= rank(dK)) and typically severely graded, so an unorthogonalized
%   block loses its trailing directions within a round or two and the sketch
%   silently returns a lower-dimensional space than it reports.  Turn it off
%   (opts.reorth = false) only to demonstrate that.
%
%   OPTS: .k (sketch width, required), .q (power rounds, default 2; total
%         applies 2q+1), .reorth (default true), .Cn (precomputed current-step
%         C, else rebuilt from P).
%
%   INFO: .k .q .ncols .ncols_raw .rank_drop .n_E_applies .n_dK_matvecs
%         .dK_nnz .dK_normF .time
%
%   dK EXACTLY ZERO (the disk_static control: the field never moves) returns an
%   empty V and zero cost, and the caller degrades to plain ILDL.  That is the
%   falsification case, not an error.
%
%   Kept LOCAL to this benchmark rather than promoted to
%   src.precond.build_deflation_V: that builder's (A, P, opts, dA) signature has
%   no room for a frozen REFERENCE factor, which is the entire input here.
%
%   See also: varvisc_esketch_ref_context, transport_V, orth_trunc,
%             src.precond.two_level_split_solve, src.precond.make_ildl_precond.

    t0 = tic;
    if nargin < 4 || isempty(opts), opts = struct(); end
    k      = opts.k;
    q      = getdef(opts, 'q',      2);
    reorth = getdef(opts, 'reorth', true);

    n  = size(K, 1);
    dK = K - ctx.Kref;                       % A_2 - A_1
    dK = (dK + dK') / 2;                     % E must be exactly symmetric;
                                             % the asymmetry is assembly round-off

    info            = struct();
    info.k          = k;
    info.q          = q;
    info.dK_nnz     = nnz(dK);
    info.dK_normF   = norm(dK, 'fro');
    info.ncols_raw  = k;

    if info.dK_nnz == 0 || k < 1
        % Nothing moved: E is exactly zero, there is no space to build, and an
        % empty coarse space is the honest answer (two_level_split_solve then
        % runs the plain ILDL split solve).
        V              = zeros(n, 0);
        Y              = zeros(n, 0);
        info.ncols     = 0;
        info.rank_drop = k;
        info.n_E_applies  = 0;
        info.n_dK_matvecs = 0;
        info.time      = toc(t0);
        return;
    end

    % E = C_ref^{-1} dK C_ref^{-T}, applied blockwise (handles are batched).
    Efun = @(X) ctx.P.applyCinv(dK * ctx.P.applyCtinv(X));

    napp = 0;                                % E applies (= dK matvecs)
    Y = randn(n, k);
    for i = 1:(2 * q + 1)
        Y    = Efun(Y);
        napp = napp + k;
        Y    = maybe_orth(Y, reorth);
    end
    if ~reorth
        Y = orth_trunc(Y);                   % an orthonormal Y is not optional:
                                             % U = C_ref^{-T} Y must not be graded
    end

    U  = ctx.P.applyCtinv(Y);                % ref-hat -> PHYSICAL subspace
    Cn = getdef(opts, 'Cn', []);
    V  = transport_V(U, P, Cn);              % orth(C_n' U): -> current hat coords

    info.ncols        = size(V, 2);
    info.rank_drop    = k - info.ncols;
    info.n_E_applies  = napp;
    info.n_dK_matvecs = napp;
    info.time         = toc(t0);
end

%==========================================================================
function Y = maybe_orth(Y, reorth)
    if reorth, Y = orth_trunc(Y); end
end

%==========================================================================
function v = getdef(s, f, d)
    if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
