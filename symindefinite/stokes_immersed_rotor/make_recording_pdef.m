function [Mfun, getU] = make_recording_pdef(Pdef, n, cap)
%MAKE_RECORDING_PDEF  Deflation coarse-operator handle that records the
%   ILDL-preconditioned residual at each MINRES iteration, for free.
%
%   [MFUN, GETU] = MAKE_RECORDING_PDEF(PDEF, N, CAP) returns a preconditioner
%   handle MFUN(r) = PDEF(r) that additionally stores r into an internal circular
%   buffer of CAP columns, together with a zero-argument getter GETU returning the
%   LAST min(#calls, CAP) recorded columns in chronological order.
%
%   Why r is the vector we want: the two-level scheme runs MINRES on the SPLIT
%   operator Ahat = C^-1 K C^-T with rhs C^-1 b (see two_level_split_solve), so the
%   vector MINRES hands to its preconditioner each iteration already IS the
%   ILDL-preconditioned residual C^-1 (b - K x) — the vector that spans the Krylov
%   subspace the solve explored.  Recording it needs no transformation and no extra
%   matvec, and because the capture is a pure side effect of the ordinary MATLAB
%   minres call, the iteration path and iteration count are UNCHANGED.
%
%   MATLAB's minres applies the 5th-argument preconditioner handle exactly once per
%   iteration (toolbox/matlab/sparfun/minres.m), so #calls ~ #iterations + 1.
%
%   The buffer keeps the LAST CAP residuals: late Lanczos vectors are the richest in
%   the directions that were slowest to converge, which are exactly the directions
%   worth deflating on the next (nearby) system.  Memory is N x CAP regardless of how
%   many iterations MINRES runs.
%
%   Columns are returned RAW — no normalization, no orthogonalization, no dedup.
%   Orthogonalization happens only when the block is turned into a deflation basis
%   (see augment_recycle_V).
%
%   Inputs:
%     Pdef - deflation coarse-operator handle in the split space, e.g.
%            src.precond.deflation_Psqrt_apply(V, Ahat2, tau, 'handle')
%     n    - problem dimension (size of the split system)
%     cap  - max # columns to keep; cap <= 0 disables recording entirely
%
%   Outputs:
%     Mfun - preconditioner handle to pass as MINRES's 5th argument
%     getU - zero-arg handle returning the recorded columns (n x k, k <= cap)
%
%   LOCAL trial version, modelled on
%   Preconditioner_Recycle/report/ball_surface_krylov_recycle/make_recording_precond.m
%   (the SPD/PCG sibling).  Promotable to +src/+precond once validated.
%
%   See also: augment_recycle_V, define_solver_list,
%             src.precond.two_level_split_solve, src.precond.deflation_Psqrt_apply.

    cap = max(0, min(round(cap), n));
    if cap == 0
        Mfun = Pdef;
        getU = @() zeros(n, 0);
        return;
    end

    buf   = zeros(n, cap);   % circular buffer over the last `cap` residuals
    nseen = 0;               % total # of preconditioner applications so far
    Mfun  = @apply;
    getU  = @getlast;

    function z = apply(r)
        nseen = nseen + 1;
        buf(:, mod(nseen - 1, cap) + 1) = r;   % r = ILDL-preconditioned residual (RAW)
        z = Pdef(r);
    end

    function U = getlast()
        k = min(nseen, cap);
        if k == 0
            U = zeros(n, 0);
            return;
        end
        % Unwind the circular buffer into chronological order (oldest kept first).
        newest = mod(nseen - 1, cap) + 1;       % column holding the most recent residual
        ord    = mod(newest - k + (0:k-1), cap) + 1;
        U      = buf(:, ord);
    end
end
