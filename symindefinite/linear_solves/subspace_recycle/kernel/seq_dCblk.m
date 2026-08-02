function [U, dC] = seq_dCblk(S, n, ref)
%SEQ_DCBLK  The rank-2nC generator pair that turns step `ref` into step `n`.
%
%   [U, dC] = SEQ_DCBLK(S, N, REF)
%
%   Returns dC = Cblk_n - Cblk_ref (n-by-nC) and the stacked generator
%   U = [dC, Sel] (n-by-2nC), so that
%
%       K_n = K_ref + U * B * U',      B = [0 I; I 0]  (2nC-by-2nC).
%
%   B is its own inverse, which is what makes the Woodbury identity in
%   lowrank_update_basis a one-liner.  range(U) is exactly the set of directions
%   the operator change can touch: any x with U'x = 0 keeps its eigenpair
%   verbatim, which is why the missing deflation component has dimension <= 2nC.
%
%   With REF == N (or on disk_static, where the coupling is constant) dC is
%   numerically zero and range(U) collapses to range(Sel) — the falsification
%   control every update variant has to reproduce.
%
%   See also: build_stokes_sequence, seq_K, lowrank_update_basis.

    if nargin < 3 || isempty(ref), ref = 1; end
    dC = S.Cblk{n} - S.Cblk{ref};
    U  = [dC, S.Sel];
end
