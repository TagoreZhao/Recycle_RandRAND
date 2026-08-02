function [C, info] = ildl_coordinate_map(P, ref)
%ILDL_COORDINATE_MAP  Materialize the ILDL split factor C and report its health.
%
%   [C, info] = ILDL_COORDINATE_MAP(P)
%   [C, info] = ILDL_COORDINATE_MAP(P, REF)
%
%   P comes from src.precond.make_ildl_precond.  Returns the explicit factor
%
%       C = S^-1 * P^T * L * |D|^{1/2},        M = C C',
%
%   the same expression build_deflation_V.m:74-76 and every test in
%   linear_solves/ rebuilds inline, plus the diagnostics needed to track how far
%   the SPLIT COORDINATE SYSTEM moves from step to step.
%
%   This matters because the deflation basis is a coordinate representation:
%   V = C' U, and deflating span(V) inside Ahat_n = C_n^-1 K_n C_n^-T actually
%   deflates span(C_n^-T C_ref' U).  If C drifts, a perfectly good V is
%   transported into nonsense even when the operator barely moved.
%
%   ldl() re-pivots on values AND pattern, and the coupling block's pattern
%   changes whenever a Lagrange point crosses a triangle edge, so p, the no-fill
%   mask on L and the 2x2 pivot placement can all change discontinuously.
%
%   Optional REF is an earlier `info` struct (or a permutation vector); when
%   given, the drift columns below are filled in.
%
%   info fields:
%     .p .nnzL .fill_ratio      permutation and factor size
%     .n_offdiag_pivots         # nonzeros below the diagonal of |D|^{1/2}, a
%                               lower bound on the number of 2x2 pivots
%     .min_absD .max_absD       extreme |D| eigenvalues (from diag(|D|^{1/2})^2)
%     .absD_ratio               max_absD / min_absD
%     .max_Disqrt               largest entry of |D|^{-1/2} (amplification)
%     .perm_hamming             # positions where p differs from REF's p (NaN if
%                               no REF); n means "completely reordered"
%     .perm_hamming_frac        the same as a fraction of n
%     .nnzL_ratio               nnzL / REF.nnzL
%
%   See also: src.precond.make_ildl_precond, transport_V, build_deflation_V.

    n    = numel(P.p);
    Sinv = spdiags(1 ./ P.s, 0, n, n);
    Pt   = sparse(P.p, (1:n)', 1, n, n);          % P^T
    C    = Sinv * Pt * P.L * P.Dsqrt;

    dsq = full(diag(P.Dsqrt)).^2;                 % |D| eigenvalues on the diagonal
    dsq = dsq(dsq > 0);
    if isempty(dsq), dsq = NaN; end

    info = struct();
    info.p                = P.p(:);
    info.nnzL             = P.nnzL;
    info.fill_ratio       = P.fill_ratio;
    info.n_offdiag_pivots = nnz(tril(P.Dsqrt, -1));
    info.min_absD         = min(dsq);
    info.max_absD         = max(dsq);
    info.absD_ratio       = max(dsq) / max(min(dsq), realmin);
    info.max_Disqrt       = full(max(abs(P.Disqrt(:))));
    info.n                = n;

    info.perm_hamming      = NaN;
    info.perm_hamming_frac = NaN;
    info.nnzL_ratio        = NaN;
    if nargin >= 2 && ~isempty(ref)
        if isstruct(ref)
            pref = ref.p(:);  nnzref = ref.nnzL;
        else
            pref = ref(:);    nnzref = NaN;
        end
        if numel(pref) == n
            info.perm_hamming      = sum(info.p ~= pref);
            info.perm_hamming_frac = info.perm_hamming / n;
        end
        info.nnzL_ratio = info.nnzL / max(nnzref, 1);
    end
end
