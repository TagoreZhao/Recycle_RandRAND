%TEST_LOWRANK_SKETCH_V  Unit tests for the low-rank A^{-1}B sketch deflation arm.
%
%   Run:  cd tests; test_lowrank_sketch_V
%
%   THE METHOD UNDER TEST.  With A_1 = K_ref factored ONCE and frozen, and
%   A_2 = K_n the current step,
%
%       D = A_1^{-1} (A_2 - A_1),   Y = (D D')^q D Omega,   V = orth(C_n' Y),
%
%   and V is handed to the existing two-level split scheme
%   (src.precond.two_level_split_solve -> deflation_Psqrt_apply on Ahat^2).
%
%   WHAT MAKES THE CLAIM CHECKABLE.  K_n - K_ref = U B U' with U = [dC, Sel] and
%   B = [0 I; I 0] invertible (build_stokes_sequence guarantees it, seq_dCblk
%   returns it), so
%
%       range(D) = K_ref^{-1} range(U),   dim <= 2*nC,
%
%   an EXACTLY known subspace of exactly known dimension.  T2/T3 therefore test the
%   sketch against that span rather than against an iteration count, and they state
%   the result over SPANS (projector residuals, both directions) rather than over a
%   chosen basis, which a randomized builder does not fix.
%
%   T5 is the falsification control: on disk_static the coupling never moves, dK is
%   exactly zero, there is no space to build, and the arm must degrade to plain ILDL
%   rather than error or invent directions.
%
%   T9 pins the REGIME the method works in, which is not "always".  A sketch narrower
%   than 2*nC keeps the directions the UPDATE moved most, and those are not the
%   directions the OPERATOR is worst conditioned in: on this fixture a truncated
%   sketch is SLOWER than the smoother alone, and only k >= 2*nC is faster.  The
%   registry default is set above that boundary because of this test, not by taste.
%
%   Uses the REAL benchmark KKT sequence (build_stokes_sequence reproduces
%   solve_stokes_immersed's assembly exactly) at h0 = 0.1, the same fixture
%   test_transport_wiring and test_exact_ldl_frozen use.  Nothing is written.

clear; clc;
here = fileparts(mfilename('fullpath'));
root = fileparts(here);                       % .../stokes_immersed_rotor
addpath(fileparts(fileparts(root)));          % repo root, for src.*
addpath(root);
addpath(fullfile(fileparts(root), 'linear_solves', 'subspace_recycle', 'kernel'));
add_recycle_paths();
rng(0);

np = 0; nf = 0;
fprintf('=== test_lowrank_sketch_V ===\n');

NSTEP = 3;
TOLV  = 1e-8;
H0    = 0.1;

S   = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', H0, ...
                                   'nsteps', NSTEP, 'quiet', true));
n   = S.n;
mit = min(2000, n);
K   = arrayfun(@(j) seq_K(S, j), 1:NSTEP, 'UniformOutput', false);
fprintf('  fixture: bar_rotating h0=%g  n=%d  nC=%d\n', H0, n, S.nC);

%% ------------------------------------------------- frozen factor apply ----
ctx = frozen_ldl_context(K{1});

% T1  the frozen factors really invert K_ref, and a multi-column right-hand side
%     gives the same answer as one column at a time.  The applier is load-bearing
%     (a wrong permutation or scaling still produces plausible-looking numbers), so
%     it is checked here rather than trusted.  The threshold scales with the
%     conditioning: relres is bounded below by cond(K_ref)*eps no matter how right
%     the applier is, while a wrong permutation gives relres >> 1 at any
%     conditioning -- hence the cap at 1.
rng(0);
Btest  = randn(n, 7);
Xbatch = frozen_ldl_apply(ctx, Btest);
Xcols  = zeros(n, 7);
for j = 1:7, Xcols(:, j) = frozen_ldl_apply(ctx, Btest(:, j)); end
relres = norm(K{1} * Xbatch - Btest, 'fro') / norm(Btest, 'fro');
kappa  = condest(K{1});
[np, nf] = chk(np, nf, sprintf('T1  frozen_ldl_apply inverts K_1 (relres %.2e, cond %.2e)', ...
                               relres, kappa), ...
    relres < min(1, max(1e-8, 50 * kappa * eps)));
[np, nf] = chk(np, nf, 'T1b batched apply == column-by-column', ...
    norm(Xbatch - Xcols, 'fro') / max(norm(Xcols, 'fro'), eps) < 1e-12);

%% ------------------------------------------ the span the sketch targets ----
STEP  = 3;                                   % a step whose coupling has moved
Kn    = K{STEP};
Pn    = src.precond.make_ildl_precond(Kn, struct('mode', 'nofill'));
Cn    = ildl_coordinate_map(Pn);

[U, dC] = seq_dCblk(S, STEP, 1);             % K_n = K_1 + U B U'
Uf      = full(U);
r_gen   = rank(Uf);                          % = rank(K_n - K_1) (B invertible)
Qref    = orth_trunc(frozen_ldl_apply(ctx, Uf));   % range(D), exactly
fprintf('  step %d: rank([dC Sel]) = %d (2*nC = %d), ||dC||_F = %.3e\n', ...
        STEP, r_gen, 2 * S.nC, norm(dC, 'fro'));

% T2  CONTAINMENT (basis-invariant).  Every direction the sketch produces lies in
%     K_1^{-1} range([dC, Sel]); k is deliberately BELOW the rank here, so this is
%     a statement about a truncated sketch, not about a lucky full-rank case.
k_small  = max(4, floor(r_gen / 2));
opts     = struct('k', k_small, 'q', 2, 'reorth', true, 'Cn', Cn);
[Vs, is_, Ys] = build_lowrank_sketch_V(ctx, Kn, Pn, opts);
[np, nf] = chk(np, nf, sprintf('T2  span(Y) is inside K_1^{-1}range([dC Sel]) (k=%d)', k_small), ...
    subspace_residual(Qref, Ys) < 1e-8);

% T3  With k at or above the rank of the generator the sketch recovers the WHOLE
%     space, in both directions -- containment alone would also hold for a single
%     lucky column.
k_full  = r_gen + 5;
[Vf, if_, Yf] = build_lowrank_sketch_V(ctx, Kn, Pn, ...
                    struct('k', k_full, 'q', 2, 'reorth', true, 'Cn', Cn));
gap = max(subspace_residual(Qref, Yf), subspace_residual(Yf, Qref));
[np, nf] = chk(np, nf, sprintf('T3  k=%d >= rank recovers the full span (gap %.2e)', k_full, gap), ...
    gap < 1e-6);
[np, nf] = chk(np, nf, sprintf('T3b rank truncation reported (%d raw -> %d cols, drop %d)', ...
                               if_.ncols_raw, if_.ncols, if_.rank_drop), ...
    if_.ncols == r_gen && if_.rank_drop == k_full - r_gen);

% T4  the contract deflation_Psqrt_apply relies on: orthonormal, no more than k
%     columns, and an SPD coarse matrix E = V' Ahat^2 V for every q (it hard-errors
%     otherwise, which is exactly the failure orth_trunc exists to prevent).
Afun  = @(y) Pn.applyCinv(Kn * Pn.applyCtinv(y));
Ahat2 = @(z) Afun(Afun(z));
spd_ok = true;
for q = [0 2]
    Vq = build_lowrank_sketch_V(ctx, Kn, Pn, ...
             struct('k', k_small, 'q', q, 'reorth', true, 'Cn', Cn));
    try
        src.precond.deflation_Psqrt_apply(Vq, Ahat2, 0.5, 'handle');
    catch
        spd_ok = false;
    end
end
[np, nf] = chk(np, nf, 'T4  V orthonormal and no wider than k', ...
    norm(Vs' * Vs - eye(size(Vs, 2)), 'fro') < 1e-12 && size(Vs, 2) <= k_small && ...
    size(Vs, 2) == min(k_small, r_gen));
[np, nf] = chk(np, nf, 'T4b V''Ahat^2 V is SPD for q = 0 and q = 2', spd_ok);

% T8  the sketch is a function of the RNG state and nothing else.
rng(0);  Va = build_lowrank_sketch_V(ctx, Kn, Pn, opts);
rng(0);  Vb = build_lowrank_sketch_V(ctx, Kn, Pn, opts);
[np, nf] = chk(np, nf, 'T8  deterministic under a fixed rng seed', isequal(Va, Vb));

%% ------------------------------------------- conditioning (the metric) ----
% T7  Iteration counts are an outcome; the conditioning is the mechanism, so both
%     are reported.  MINRES is handed the Psqrt HANDLE as its 5th argument, which
%     MATLAB applies as M^{-1}: the operator it actually sees is G*Ahat (captured
%     modes at +-sqrt(tau)), so that -- not G*Ahat*G -- is the spectrum to measure.
%     lambda_min / lambda_max are reported separately, not folded into kappa alone.
if n <= 3000
    tau  = 0.5;
    Ahat = full(Pn.applyCinv(Kn * Pn.applyCtinv(eye(n))));
    Ahat = (Ahat + Ahat') / 2;
    G    = src.precond.deflation_Psqrt_apply(Vf, Ahat2, tau, 'matrix');
    ev0  = sort(abs(eig(Ahat)));
    ev1  = sort(abs(eig(G * Ahat)));
    k0   = ev0(end) / ev0(1);
    k1   = ev1(end) / ev1(1);
    fprintf(['    |lambda| range: Ahat [%.3e, %.3e] kappa %.3e -> ' ...
             'G*Ahat [%.3e, %.3e] kappa %.3e\n'], ...
            ev0(1), ev0(end), k0, ev1(1), ev1(end), k1);
    [np, nf] = chk(np, nf, sprintf('T7  deflation improves conditioning (%.2e -> %.2e)', k0, k1), ...
        k1 < k0);
else
    fprintf('  SKIP T7 (n = %d > 3000; dense spectrum too large)\n', n);
end

%% ------------------------------------------------- the registry closure ----
% k = 2 * 25 = 50, just above the rank of the generator on this fixture (2*nC = 48),
% which is the regime T9 below shows is the one that works.
params = struct('LOWRANK_SM_EIG', 25, 'LOWRANK_OVERSAMPLE', 2, ...
                'LOWRANK_SKETCH_Q', 2, 'LOWRANK_REF_REFRESH', Inf, ...
                'ILDL_PREC_REFRESH', 1, 'DEFLAT_TAU', 0.5);
solvers = define_solver_list(params);
skeys   = cellfun(@(s) s.key, solvers, 'UniformOutput', false);
[np, nf] = chk(np, nf, 'T6  two_level_lowrank_sketch is registered', ...
    any(strcmp(skeys, 'two_level_lowrank_sketch')));
sl = solvers{strcmp(skeys, 'two_level_lowrank_sketch')};
si = solvers{strcmp(skeys, 'ildl_nofill')};

% ONE shared pc, exactly as solve_stokes_immersed drives it, so both arms use the
% same ILDL smoother object and differ only by the coarse space.
pc = struct('step', 1, 'nC', S.nC, ...
            'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
its_l = zeros(NSTEP, 1);  fls = zeros(NSTEP, 1);  err_l = zeros(NSTEP, 1);
its_i = zeros(NSTEP, 1);  err_i = zeros(NSTEP, 1);
lastwarn('');
for j = 1:NSTEP
    pc.step = j;  pc.K = K{j};
    [xj, fls(j), ~, its_l(j)] = sl.solve(K{j}, S.b{j}, TOLV, mit, pc);
    err_l(j) = norm(S.b{j} - K{j} * xj) / norm(S.b{j});
    [xi, ~, ~, its_i(j)] = si.solve(K{j}, S.b{j}, TOLV, mit, pc);
    err_i(j) = norm(S.b{j} - K{j} * xi) / norm(S.b{j});
end
fprintf('    iters  sketch %s  vs ILDL %s\n', mat2str(its_l(:)'), mat2str(its_i(:)'));
fprintf('    true relres  sketch %.2e  vs ILDL %.2e (split relres met %.0e)\n', ...
        max(err_l), max(err_i), TOLV);

% T6b the engine contract: a SCALAR iteration count (the engine assigns it into one
%     element of a per-step array) and convergence.  ACCURACY is measured against the
%     ILDL arm rather than against a constant: two_level_split_solve's relres is the
%     SPLIT system's, so the true residual sits a smoother-conditioning factor above
%     the requested tolerance for BOTH arms (see two_level_it.m).  What must hold is
%     that this arm is no less accurate than the one it shares a smoother with.
[np, nf] = chk(np, nf, 'T6b scalar iteration count, converged, accuracy on par with ILDL', ...
    all(arrayfun(@(v) isscalar(v), its_l)) && all(fls == 0) && ...
    max(err_l) <= 10 * max(err_i));

% T6c the recycled object is the FACTORIZATION: built at step 1, never rebuilt, and
%     never silently un-frozen by the shape guard.
e = pc.cache('lowrank_ref');
[np, nf] = chk(np, nf, 'T6c reference factorization frozen at step 1', ...
    e.step == 1 && e.val.n == n && isempty(lastwarn));

% T6d the coarse space width follows k = oversample * sm_eig, capped by the rank of
%     the generator -- the number to quote in any comparison is the EFFECTIVE one.
ei = pc.cache('lowrank_info');
[np, nf] = chk(np, nf, sprintf('T6d k = 2*25 = 50 -> %d effective columns (rank cap %d)', ...
                               ei.val.ncols, r_gen), ...
    ei.val.k == 50 && ei.val.ncols == min(50, r_gen) && ei.step == NSTEP);

% T6e step 1 is the CONTROLLED comparison: dK = 0 there, so the arm IS the ILDL arm
%     and the counts must be identical, not merely close.
[np, nf] = chk(np, nf, sprintf('T6e step 1 (dK = 0) identical to ILDL (%d its)', its_l(1)), ...
    its_l(1) == its_i(1));

%% -------------------------------------- the regime the method works in ----
% T9  THE BOUNDARY, pinned rather than assumed.  range(D) has dimension 2*nC, so a
%     sketch narrower than that keeps only the directions the UPDATE moved most --
%     which is not the same as the directions the OPERATOR is worst conditioned in.
%     Measured on this fixture: k = 15 needs MORE iterations than the smoother alone,
%     k >= 2*nC needs substantially fewer.  Deflation being "never worse" is
%     conditional, and this is the condition for this arm.
STEP2 = 2;
P2 = src.precond.make_ildl_precond(K{STEP2}, struct('mode', 'nofill'));
C2 = ildl_coordinate_map(P2);
mk = @(kk) build_lowrank_sketch_V(ctx, K{STEP2}, P2, ...
               struct('k', kk, 'q', 2, 'reorth', true, 'Cn', C2));
[~, ~, ~, it_none]  = src.precond.two_level_split_solve(K{STEP2}, S.b{STEP2}, ...
                                                        TOLV, mit, P2, [], 0.5);
[~, ~, ~, it_trunc] = src.precond.two_level_split_solve(K{STEP2}, S.b{STEP2}, ...
                                                        TOLV, mit, P2, mk(15), 0.5);
[~, ~, ~, it_full]  = src.precond.two_level_split_solve(K{STEP2}, S.b{STEP2}, ...
                                                        TOLV, mit, P2, mk(r_gen + 5), 0.5);
fprintf('    step %d: ILDL only %d | k=15 (truncated) %d | k=%d (full span) %d\n', ...
        STEP2, it_none, it_trunc, r_gen + 5, it_full);
[np, nf] = chk(np, nf, sprintf('T9  full span beats the smoother alone (%d < %d its)', ...
                               it_full, it_none), ...
    it_full < it_none);
[np, nf] = chk(np, nf, sprintf('T9b full span beats a truncated sketch (%d < %d its)', ...
                               it_full, it_trunc), ...
    it_full < it_trunc);

%% ------------------------------------------ falsification control: static ----
% T5  disk_static never moves the coupling, so dK is exactly zero, D is exactly
%     zero and there is NO space to build.  The arm must then be plain ILDL --
%     identical iteration counts, not merely similar ones.
Ss  = build_stokes_sequence(struct('case_name', 'disk_static', 'h0', H0, ...
                                   'nsteps', 2, 'quiet', true));
Ks  = seq_K(Ss, 2);
ctxs = frozen_ldl_context(seq_K(Ss, 1));
Ps  = src.precond.make_ildl_precond(Ks, struct('mode', 'nofill'));
[V0, i0] = build_lowrank_sketch_V(ctxs, Ks, Ps, ...
               struct('k', 10, 'q', 2, 'reorth', true, 'Cn', ildl_coordinate_map(Ps)));

pcs = struct('step', 1, 'nC', Ss.nC, ...
             'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
mits = min(2000, Ss.n);
[~, ~, ~, it_l1] = sl.solve(seq_K(Ss, 1), Ss.b{1}, TOLV, mits, pcs);
[~, ~, ~, it_i1] = si.solve(seq_K(Ss, 1), Ss.b{1}, TOLV, mits, pcs);
pcs.step = 2;
[~, ~, ~, it_l2] = sl.solve(Ks, Ss.b{2}, TOLV, mits, pcs);
[~, ~, ~, it_i2] = si.solve(Ks, Ss.b{2}, TOLV, mits, pcs);

[np, nf] = chk(np, nf, sprintf('T5  disk_static: dK = 0 -> empty V (%d cols, nnz(dK) %d)', ...
                               i0.ncols, i0.dK_nnz), ...
    isempty(V0) && i0.ncols == 0 && i0.dK_nnz == 0 && i0.n_backsolves == 0);
[np, nf] = chk(np, nf, sprintf('T5b disk_static: falls back to ILDL exactly (%d/%d vs %d/%d its)', ...
                               it_l1, it_l2, it_i1, it_i2), ...
    it_l1 == it_i1 && it_l2 == it_i2);

%% ------------------------------------------------------------ summary ----
fprintf('\n%d passed, %d FAILED\n', np, nf);
if nf > 0, error('test_lowrank_sketch_V:failures', '%d checks failed', nf); end

%==========================================================================
function s = subspace_residual(Qbase, Y)
%SUBSPACE_RESIDUAL  sin of the largest principal angle of span(Y) into span(Qbase).
% Basis-invariant: depends on the two SPANS only, not on how either is written.
% 0 means span(Y) is contained in span(Qbase); 1 means a direction is orthogonal
% to it.  Both arguments are orthonormalized here so callers may pass raw blocks.
    Q = orth_trunc(Qbase);
    Z = orth_trunc(Y);
    if isempty(Z), s = 0; return; end
    if isempty(Q), s = 1; return; end
    s = norm(Z - Q * (Q' * Z));
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;  fprintf('  ok   %s\n', name);
    else
        nf = nf + 1;  fprintf('  FAIL %s\n', name);
    end
end
