function K = seq_K(S, n)
%SEQ_K  Materialize the step-n KKT matrix from the sequence's low-rank form.
%
%   K = SEQ_K(S, N)  with S from build_stokes_sequence.
%
%   The whole time sequence is a rank-<= 2*nC update of ONE fixed matrix:
%
%       K_n = K0 + Cblk_n * Sel' + Sel * Cblk_n'
%
%   where K0 is the KKT assembled with a ZERO coupling block (fluid blocks +
%   Dirichlet/pin elimination, all time-independent), Sel = [0; 0; I_nC] selects
%   the multiplier rows, and Cblk_n = [Z_u * C(t_n)'; 0; 0] carries the moving
%   coupling (Z_u = the time-independent velocity-Dirichlet mask).
%
%   Consequently, for ANY pair of steps m, n
%
%       K_n = K_m + dC * Sel' + Sel * dC',      dC = Cblk_n - Cblk_m
%
%   which is what makes a frozen factorization of K_m an exact (Woodbury)
%   inverse for K_n and bounds the eigenspace motion by rank 2*nC.
%   build_stokes_sequence verifies this identity against the directly assembled
%   matrix at construction time (opts.verify).
%
%   Storing the sequence in this form rather than as 60 explicit sparse matrices
%   keeps the cache file small; the reconstruction is one sparse rank-2nC add.
%
%   See also: build_stokes_sequence, seq_dCblk, lowrank_update_basis.

    if n < 1 || n > S.nsteps
        error('seq_K:badStep', 'step %d out of range 1..%d', n, S.nsteps);
    end
    W = S.Cblk{n} * S.Sel';
    K = S.K0 + W + W';
end
