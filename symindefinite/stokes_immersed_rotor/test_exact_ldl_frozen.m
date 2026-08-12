% TEST_EXACT_LDL_FROZEN  The exact-LDL preconditioner and the frozen registry arm.
%
% WHAT THIS PINS DOWN.  Every other arm in define_solver_list uses an APPROXIMATE
% smoother rebuilt every step (ILDL_PREC_REFRESH = 1), so its per-step iteration
% count mixes two effects: how good the approximation is, and how far the KKT
% matrix drifted.  The exact-LDL arm removes the first.  With no dropping,
% C = S^-1 P^T L |D|^{1/2} satisfies C C^T = |K| exactly and the split operator is
%
%     Ahat = C^-1 K C^-T = |D|^{-1/2} D |D|^{-1/2} = sign(D),
%
% whose spectrum is exactly {+1,-1}: 2 MINRES iterations on the matrix it was
% built from.  Built once at step 1 and frozen (EXACT_PREC_REFRESH = Inf), every
% iteration above 2 at step n is therefore PURE DRIFT of the coupling block C(t_n).
%
% Uses the REAL benchmark KKT sequence (build_stokes_sequence reproduces
% solve_stokes_immersed's assembly exactly), same fixture recipe as
% test_transport_wiring, because the sparsity pattern has to actually move.
%
% Run:  test_exact_ldl_frozen

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
repoRoot = fileparts(fileparts(thisFileDir));      % .../Recycle_RandRAND
addpath(repoRoot);
addpath(fullfile(repoRoot, 'symindefinite', 'linear_solves', ...
                 'subspace_recycle', 'kernel'));
add_recycle_paths();
rng(1);

NSTEP = 3;
tolv  = 1e-8;
H0    = 0.1;      % the coarse twin of the benchmark mesh; keeps the test fast

S   = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', H0, ...
                                   'nsteps', NSTEP, 'quiet', true));
n   = S.n;
mit = min(2000, n);
K   = arrayfun(@(j) seq_K(S, j), 1:NSTEP, 'UniformOutput', false);

fprintf('=== test_exact_ldl_frozen (n=%d, nC=%d) ===\n', n, S.nC);
npass = 0;

%% ---- T1: 'exact' mode drops nothing ------------------------------------
Pex = src.precond.make_ildl_precond(K{1}, struct('mode', 'exact'));
Pnf = src.precond.make_ildl_precond(K{1}, struct('mode', 'nofill'));

assert(strcmp(Pex.mode, 'exact'), 'T1: mode not recorded as "exact" (got "%s")', Pex.mode);
assert(Pex.nnzL > Pnf.nnzL, ...
       'T1: the exact factor has no more nonzeros than the no-fill one (%d vs %d)', ...
       Pex.nnzL, Pnf.nnzL);
% Compared against nofill, not against 1: the level-0 mask is
% spones(tril(Aperm,-1)) + speye(n), and K has structurally-zero diagonal rows in
% the constraint block, so Pnf.fill_ratio is already slightly above 1.
assert(Pex.fill_ratio > Pnf.fill_ratio, ...
       'T1: fill_ratio did not grow (exact %.3f vs nofill %.3f)', ...
       Pex.fill_ratio, Pnf.fill_ratio);
fprintf('  PASS T1: nnzL %d -> %d, fill_ratio %.3f -> %.3f\n', ...
        Pnf.nnzL, Pex.nnzL, Pnf.fill_ratio, Pex.fill_ratio);
npass = npass + 1;

%% ---- T2: Ahat^2 = I, i.e. the spectrum really is +-1 -------------------
% The strong assertion.  Tolerance is 1e-8 rather than eps because Ahat(Ahat(x))
% is FOUR triangular solves with an exact but ill-conditioned factor; the
% discriminating power comes from the negative control below, which is O(1).
Ahat_ex = @(y) Pex.applyCinv(K{1} * Pex.applyCtinv(y));
Ahat_nf = @(y) Pnf.applyCinv(K{1} * Pnf.applyCtinv(y));

err_ex = 0;  err_nf = 0;
for j = 1:5
    x = randn(n, 1);
    err_ex = max(err_ex, norm(Ahat_ex(Ahat_ex(x)) - x) / norm(x));
    err_nf = max(err_nf, norm(Ahat_nf(Ahat_nf(x)) - x) / norm(x));
end
assert(err_ex < 1e-8, 'T2: Ahat^2 is not the identity (rel err %.3e)', err_ex);
assert(err_nf > 1e-2, ...
       ['T2 NEGATIVE CONTROL FAILED: the no-fill factor already squares to I ' ...
        '(rel err %.3e) -- T2 proves nothing on this matrix'], err_nf);
fprintf('  PASS T2: ||Ahat^2 x - x||/||x|| = %.2e (exact) vs %.2e (no-fill control)\n', ...
        err_ex, err_nf);
npass = npass + 1;

%% ---- T3: 2 iterations on the matrix it was built from -------------------
[~, fl_ex, ~, it_ex] = src.precond.two_level_split_solve( ...
                           K{1}, S.b{1}, tolv, mit, Pex, [], 0.5);
[~, fl_nf, ~, it_nf] = src.precond.two_level_split_solve( ...
                           K{1}, S.b{1}, tolv, mit, Pnf, [], 0.5);
assert(fl_ex == 0, 'T3: the exact-preconditioned solve did not converge (flag %d)', fl_ex);
assert(it_ex <= 3, 'T3: exact factor took %d iterations, expected <= 3', it_ex);
assert(it_ex < it_nf, 'T3: exact (%d its) is not faster than no-fill (%d its)', it_ex, it_nf);
fprintf('  PASS T3: fresh exact factor solves in %d its (no-fill needs %d, flag %d)\n', ...
        it_ex, it_nf, fl_nf);
npass = npass + 1;

%% ---- T4: registry wiring, position, and the freeze ---------------------
params  = struct('EXACT_PREC_REFRESH', Inf, 'DEFLAT_SM_EIG', 20, ...
                 'DEFLAT_RECYCLE_K', 0, 'ILDL_PREC_REFRESH', 1);
solvers = define_solver_list(params);
skeys   = cellfun(@(s) s.key, solvers, 'UniformOutput', false);

% A LOWER BOUND, not an equality: what this test is about is that exact_ldl_frozen
% sits at position 4 (checked next), and the registry is meant to grow -- pinning the
% exact count made an unrelated new arm look like a regression in this file.
assert(numel(skeys) >= 9, 'T4: registry has %d entries, expected at least 9', numel(skeys));
assert(isequal(skeys(1:4), {'minres_unprec'; 'block_jacobi'; 'ildl_nofill'; ...
                            'exact_ldl_frozen'}), ...
       'T4: exact_ldl_frozen is not at registry position 4 (got %s)', ...
       strjoin(skeys(1:4)', ', '));

sx = solvers{strcmp(skeys, 'exact_ldl_frozen')};

% One shared pc, exactly as +src/+stokes/solve_stokes_immersed drives it.
pc  = struct('step', 1, 'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
its = zeros(NSTEP, 1);
for j = 1:NSTEP
    pc.step = j;
    [x, fl, ~, its(j)] = sx.solve(K{j}, S.b{j}, tolv, mit, pc);
    assert(fl == 0, 'T4: step %d did not converge (flag %d)', j, fl);
    rel = norm(K{j}*x - S.b{j}) / norm(S.b{j});
    assert(rel < 1e-6, 'T4: step %d residual %.3e is too large', j, rel);
end

e = pc.cache('exact_ldl_frozen');
assert(e.step == 1, ...
       'T4: the factor was rebuilt at step %d, but EXACT_PREC_REFRESH = Inf', e.step);
assert(~any(startsWith(keys(pc.cache), 'ildl_')), ...
       'T4: the arm wrote an ildl_* cache key -- it can alias the ILDL smoother');
assert(its(1) <= 3, 'T4: step 1 took %d iterations, expected <= 3', its(1));
assert(its(2) > its(1), ...
       ['T4: step 2 did not cost more than step 1 (%d vs %d) -- the frozen factor ' ...
        'is not measuring drift'], its(2), its(1));
fprintf('  PASS T4: position 4 of 9, factor built at step %d only, iters %s\n', ...
        e.step, mat2str(its(:)'));
npass = npass + 1;

%% ---- T5: the cache key cannot alias the ILDL smoother ------------------
% Even when a caller sets ILDL_MODE = 'exact' -- which makes the ILDL/two-level
% arms take the key 'ildl_exact' -- the frozen arm keeps its own entry under its
% own cadence.  Every key two_level_parts can emit starts with 'ildl_'.
p5  = struct('EXACT_PREC_REFRESH', Inf, 'ILDL_MODE', 'exact', ...
             'DEFLAT_SM_EIG', 20, 'DEFLAT_RECYCLE_K', 0, 'ILDL_PREC_REFRESH', 1);
s5  = define_solver_list(p5);
k5  = cellfun(@(s) s.key, s5, 'UniformOutput', false);
sx5 = s5{strcmp(k5, 'exact_ldl_frozen')};
si5 = s5{strcmp(k5, 'ildl_nofill')};

pc5 = struct('step', 1, 'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
[~, f5a] = sx5.solve(K{1}, S.b{1}, tolv, mit, pc5);
[~, f5b] = si5.solve(K{1}, S.b{1}, tolv, mit, pc5);
assert(f5a == 0 && f5b == 0, 'T5: a solve failed (flags %d/%d)', f5a, f5b);
ck = keys(pc5.cache);
assert(any(strcmp(ck, 'exact_ldl_frozen')) && any(strcmp(ck, 'ildl_exact')), ...
       'T5: the two factors did not get separate cache entries (keys: %s)', ...
       strjoin(ck, ', '));
fprintf('  PASS T5: separate cache entries under ILDL_MODE=''exact'' (%s)\n', ...
        strjoin(sort(ck), ', '));
npass = npass + 1;

%% ---- T6: the frozen factor refuses to be applied to a different size ---
% size(K,1) = nU + nP + nC, and nC drops if a Lagrange point leaves the fluid
% mesh.  A factor frozen at step 1 is APPLIED at every later step, so without a
% reuse predicate this is an opaque dimension error inside applyCinv's `s .* r`.
% Here we drop the last two multipliers, which is exactly that situation.
m  = n - 2;
Ks = K{2}(1:m, 1:m);
bs = S.b{2}(1:m);
lastwarn('', '');
pc.step = NSTEP + 1;
warnState = warning('off', 'define_solver_list:cacheShapeChanged');
cleanup = onCleanup(@() warning(warnState));
[~, fl6, ~, it6] = sx.solve(Ks, bs, tolv, min(2000, m), pc);
clear cleanup;
[~, wid] = lastwarn;

assert(strcmp(wid, 'define_solver_list:cacheShapeChanged'), ...
       'T6: the shape change was not warned about (last warning id "%s")', wid);
e6 = pc.cache('exact_ldl_frozen');
assert(e6.step == NSTEP + 1 && numel(e6.val.s) == m, ...
       'T6: the cache was not rebuilt for the new size (step %d, n %d, expected %d)', ...
       e6.step, numel(e6.val.s), m);
assert(fl6 == 0 && it6 <= 3, ...
       'T6: the rebuilt factor is not exact for the new system (flag %d, %d its)', ...
       fl6, it6);
fprintf('  PASS T6: shape change warned + rebuilt (n %d -> %d, %d its)\n', n, m, it6);
npass = npass + 1;

%% ---- T7: control -- a constant K must stay at 2 iterations -------------
% disk_static never moves, so K is the same matrix at every step.  If the freeze
% works, the frozen exact factor stays exact.  T7 failing while T3 passes means
% the freeze is broken, not the theory.
Sd = build_stokes_sequence(struct('case_name', 'disk_static', 'h0', H0, ...
                                  'nsteps', NSTEP, 'quiet', true));
Kd = arrayfun(@(j) seq_K(Sd, j), 1:NSTEP, 'UniformOutput', false);
dK = norm(Kd{NSTEP} - Kd{1}, 'fro') / norm(Kd{1}, 'fro');
assert(dK < 1e-12, ...
       'T7: disk_static K is not constant (rel change %.3e) -- this control is void', dK);

pcd  = struct('step', 1, 'cache', containers.Map('KeyType', 'char', 'ValueType', 'any'));
itsd = zeros(NSTEP, 1);
for j = 1:NSTEP
    pcd.step = j;
    [~, fld, ~, itsd(j)] = sx.solve(Kd{j}, Sd.b{j}, tolv, min(2000, Sd.n), pcd);
    assert(fld == 0, 'T7: step %d did not converge (flag %d)', j, fld);
end
assert(all(itsd <= 3), ...
       'T7: the frozen factor drifted on a CONSTANT matrix (iters %s)', mat2str(itsd(:)'));
fprintf('  PASS T7: constant-K control stays at %s iterations\n', mat2str(itsd(:)'));
npass = npass + 1;

fprintf('=== test_exact_ldl_frozen: %d checks passed ===\n', npass);
