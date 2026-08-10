function params = make_woodbury_params()
%MAKE_WOODBURY_PARAMS  Default parameters for the Woodbury-update benchmark.
%   PARAMS = MAKE_WOODBURY_PARAMS()
%
%   SIZING.  h0 = 0.05, dt = 0.02, Tstep = 61 are copied VERBATIM from the parent
%   benchmark's run_benchmark.m so this study's numbers sit next to that study's
%   iteration counts without a size caveat.  ntot = 2N + N + nC ~ 5120.
%
%   THE REFERENCE IS HARD-FROZEN AT STEP 1 and there is deliberately no refresh
%   cadence knob.  The question this study asks is "can ONE exact factorization of
%   A_1 serve the whole sequence?", and a refresh knob would let that question be
%   answered by refactorizing.  Re-anchoring is a follow-up study, not a parameter.
%
%   See also: solve_woodbury_sequence, run_woodbury_benchmark, woodbury_solve.

    params = struct();

    % --- Discretization (identical to the parent KKT benchmark) -------------
    params.h0    = 0.03;
    params.dt    = 0.02;
    params.Tstep = 61;                  % => 60 solves; also sets Tmax = dt*Tstep

    % --- Cases --------------------------------------------------------------
    % disk_static is NOT filler: its coupling block is constant, so dC is exactly
    % zero and the Woodbury correction must collapse to the frozen inverse
    % bit-for-bit.  It is the falsification control for the whole scheme.
    params.cases = {'bar_rotating', 'disk_translating', 'disk_static'};

    % --- Run length ---------------------------------------------------------
    % Trim the RUN with max_steps, never with Tstep: Tstep sets Tmax and hence the
    % rotor's angular velocity, so shrinking it would change the geometry under
    % test instead of just doing fewer solves.
    params.max_steps = [];

    % --- Sequence construction ---------------------------------------------
    % build_stokes_sequence asserts K_n = K0 + Cblk_n*Sel' + Sel*Cblk_n' to <1e-12
    % at every step.  That identity is the premise of the entire method, so it is
    % verified on every run rather than trusted from the cache metadata.
    params.verify_lowrank = true;
    params.use_cache      = false;       % kernel/cache/, already gitignored

    % --- Timing -------------------------------------------------------------
    % Wall clock is a headline metric here (both arms are direct solves, so
    % iteration counts do not exist), which means it has to be measured honestly:
    % each solve is repeated and the MINIMUM taken, because the max and the mean
    % both measure the OS scheduler as much as the algorithm.
    params.TIME_REPEATS = 3;

    % --- Diagnostics --------------------------------------------------------
    % kappa(Cap) is ALWAYS computed: Cap is 2nC-by-2nC dense (40 or 88), so its
    % full SVD is free, and it is the quantity the Woodbury forward error tracks.
    %
    % kappa(K_n) is off by default: at ntot ~ 5120 it needs eigs at both ends of
    % the spectrum, which costs more than the benchmark it would annotate.  When
    % turning it on, note that eigs SILENTLY IGNORES an options struct -- pass
    % name-value pairs ('Tolerance', ...) or the symmetric mode is lost.
    params.COMPUTE_SPECTRUM = false;
    params.SPECTRUM_TOL     = 1e-8;     % eigs tolerance when the above is true
end
