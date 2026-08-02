% TEST_TWO_LEVEL_IT  Unit tests for the instrumented two-level split solve.
%
% T1 is the non-intrusiveness contract: the instrumented path must reproduce
% src.precond.two_level_split_solve exactly, so every number this study reports
% is a number the production scheme would have produced.
%
% T2 pins the folder convention that the split-operator relres is NOT the true
% residual — the study reports true_res everywhere.
%
% Run:  test_two_level_it

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
add_recycle_paths();
rng(1);

S = build_stokes_sequence(struct('case_name', 'bar_rotating', 'h0', 0.15, ...
                                 'nsteps', 2, 'use_cache', false, 'quiet', true));
K = seq_K(S, 1);  b = S.b{1};  n = S.n;  k = 25;  tau = 1;
P = src.precond.make_ildl_precond(K, struct('mode', 'nofill'));
C = ildl_coordinate_map(P);
M = C * C';  M = (M + M') / 2;
[U, ~] = eigs(K, M, k, 'smallestabs', 'Tolerance', 1e-11, 'MaxIterations', 2000);
V = transport_V(U, P, C);
so = struct('tau', tau, 'tol', 1e-8, 'maxit', n);

fprintf('=== test_two_level_it (n=%d, k=%d, tau=%g) ===\n', n, k, tau);
npass = 0;

%% ---- T1: identical to the production path -----------------------------
res = two_level_it(K, b, P, V, so);
[xp, flp, rrp, itp] = src.precond.two_level_split_solve(K, b, 1e-8, n, P, V, tau);
assert(res.iters == itp, 'T1: iteration count differs (%d vs %d)', res.iters, itp);
assert(res.flag == flp, 'T1: flag differs');
assert(abs(res.relres - rrp) <= 1e-14 * max(rrp, eps), 'T1: relres differs');
assert(norm(res.x - xp) <= 1e-12 * max(norm(xp), eps), 'T1: solution differs');
fprintf('  PASS T1: matches two_level_split_solve exactly (%d iters)\n', res.iters);
npass = npass + 1;

%% ---- T2: true residual is reported and differs from relres ------------
% MINRES drives the SPLIT residual below tol; the true residual comes back
% amplified by the conditioning of C — here by ~100x, which is exactly why this
% folder never reports the solver's own relres.  The check is on the
% amplification being real and bounded, not on an arbitrary absolute cutoff.
tr = norm(b - K * res.x) / norm(b);
assert(abs(res.true_res - tr) < 1e-14, 'T2: true_res mis-computed');
assert(res.flag == 0, 'T2: MINRES did not converge (flag %d)', res.flag);
assert(res.relres <= 1e-8 * (1 + 1e-6), 'T2: split relres above tol (%.3e)', res.relres);
amp = res.true_res / res.relres;
assert(res.true_res < 1e-4, 'T2: true residual implausibly large (%.3e)', res.true_res);
assert(amp > 2, ['T2: expected the split relres to understate the true residual ' ...
                 '(true %.2e vs relres %.2e)'], res.true_res, res.relres);
fprintf('  PASS T2: flag=0, split relres=%.2e, true_res=%.2e (amplification %.0fx)\n', ...
        res.relres, res.true_res, amp);
npass = npass + 1;

%% ---- T3: the V = [] baseline ------------------------------------------
r0 = two_level_it(K, b, P, [], so);
assert(r0.coarse_dim == 0 && isnan(r0.condE), 'T3: baseline should report no coarse space');
assert(r0.flag == 0 && r0.true_res < 1e-6, 'T3: ILDL-only solve failed');
assert(r0.iters > res.iters, ...
       'T3: deflation should help at step 1 (ildl %d vs two-level %d)', ...
       r0.iters, res.iters);
fprintf('  PASS T3: ILDL-only %d iters vs two-level %d (%.2fx)\n', ...
        r0.iters, res.iters, r0.iters/res.iters);
npass = npass + 1;

%% ---- T4: coarse health matches a dense computation --------------------
Ahat  = @(Y) P.applyCinv(K * P.applyCtinv(Y));
E_ref = V' * Ahat(Ahat(V));
E_ref = (E_ref + E_ref') / 2;
ee    = sort(real(eig(E_ref)), 'ascend');
assert(abs(res.minEigE - ee(1)) <= 1e-8 * abs(ee(end)), 'T4: minEigE mismatch');
assert(abs(res.maxEigE - ee(end)) <= 1e-8 * abs(ee(end)), 'T4: maxEigE mismatch');
assert(abs(res.sqrt_minEigE - sqrt(max(ee(1), 0))) < 1e-10, 'T4: sqrt_minEigE mismatch');
fprintf('  PASS T4: cond(E)=%.2e, sqrt(min eig E)=%.2e matches dense eig\n', ...
        res.condE, res.sqrt_minEigE);
npass = npass + 1;

%% ---- T5: the recording tap is free and non-intrusive ------------------
so_rec = so;  so_rec.record = 40;
rr_ = two_level_it(K, b, P, V, so_rec);
assert(rr_.iters == res.iters, ...
       'T5: recording tap changed the iteration path (%d vs %d)', rr_.iters, res.iters);
assert(norm(rr_.x - res.x) <= 1e-12 * max(norm(res.x), eps), ...
       'T5: recording tap changed the solution');
assert(size(rr_.W, 2) == min(40, res.iters) && all(isfinite(rr_.W(:))), ...
       'T5: captured block has the wrong shape (%d cols)', size(rr_.W, 2));
assert(isempty(res.W) || size(res.W, 2) == 0, 'T5: record=0 should capture nothing');
fprintf('  PASS T5: record=40 captured %d x %d, iteration path unchanged\n', ...
        size(rr_.W, 1), size(rr_.W, 2));
npass = npass + 1;

fprintf('=== test_two_level_it: %d checks passed ===\n', npass);
