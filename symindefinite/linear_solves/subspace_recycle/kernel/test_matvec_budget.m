% TEST_MATVEC_BUDGET  Unit tests for the work-unit model.
%
% The model exists to make one specific claim checkable: at the benchmark's
% k=500 the coarse correction, applied every MINRES iteration, dwarfs the
% operator apply — so an iteration-count win can still be a wall-clock loss.
% T4 asserts that crossover exists and locates it.
%
% Run:  test_matvec_budget

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(thisFileDir);
add_recycle_paths();

fprintf('=== test_matvec_budget ===\n');
npass = 0;

% Benchmark-scale numbers (n=5840, nnz from the h0=0.05 KKT).
base = struct('n', 5840, 'nnzK', 60000, 'nnzL', 30000);

%% ---- T1: ILDL-only costs exactly one unit per iteration ---------------
s = base;  s.k = 0;  s.iters = 312;
[u, d] = matvec_budget(s);
assert(abs(u - 312) < 1e-9, 'T1: expected 312 units, got %g', u);
assert(d.coarse_apply == 0 && d.setup == 0, 'T1: baseline should have no coarse cost');
fprintf('  PASS T1: ILDL-only 312 iters -> %.0f units (1 per iteration)\n', u);
npass = npass + 1;

%% ---- T2: monotone in k and in iterations ------------------------------
prev = -inf;
for k = [0 25 50 100 250 500]
    s = base;  s.k = k;  s.iters = 100;  s.build_E = true;
    u = matvec_budget(s);
    assert(u > prev, 'T2: total not increasing in k at k=%d', k);
    prev = u;
end
s1 = base; s1.k = 100; s1.iters = 100;
s2 = base; s2.k = 100; s2.iters = 200;
assert(matvec_budget(s2) > matvec_budget(s1), 'T2: not increasing in iterations');
fprintf('  PASS T2: monotone in k and in iteration count\n');
npass = npass + 1;

%% ---- T3: the terms add up ---------------------------------------------
s = base;  s.k = 500;  s.iters = 100;  s.build_E = true;  s.qr_cols = 500;
s.n_backsolves = 20;  s.n_ldl = 1;  s.n_eigs_k = 1;
[u, d] = matvec_budget(s);
assert(abs(d.setup + d.iter - d.total) < 1e-6 * d.total, 'T3: setup+iter != total');
assert(abs(u - d.total) < 1e-9, 'T3: units != detail.total');
assert(abs(d.iter - (d.operator + d.coarse_apply)) < 1e-6 * d.iter, 'T3: iter split');
fprintf('  PASS T3: setup %.0f + iter %.0f = total %.0f units\n', d.setup, d.iter, d.total);
npass = npass + 1;

%% ---- T4: the E rebuild, not the coarse apply, is what breaks the budget -
% Measured on this machine at n=5840, k=500: the coarse apply is 3.6x one
% operator apply and E = V'Ahat^2 V costs 546 units, against a 312-unit
% ILDL-only budget.  The model must reproduce the ORDER of both, otherwise a
% wall-clock verdict drawn from it is worthless.
base.nnzK = 83540;  base.nnzL = 44571;      % the real h0=0.05 KKT

sk = base;  sk.k = 500;  sk.iters = 1;
[~, dk] = matvec_budget(sk);
ratio = dk.coarse_apply / dk.operator;
assert(ratio > 1 && ratio < 12, ...
       ['T4: k=500 coarse apply modelled at %.1fx the operator apply; measured ' ...
        '3.6x.  A flop-only count gives ~68x — check dense_speedup.'], ratio);

sE = base;  sE.k = 500;  sE.iters = 0;  sE.build_E = true;
[~, dE] = matvec_budget(sE);
assert(abs(dE.coarse_build - 1000) < 1, 'T4: E build should be 2k = 1000 units');
assert(dE.coarse_build > 312, ...
       'T4: the k=500 E rebuild should alone exceed the 312-unit ILDL budget');

% Break-even in k, at the oracle's own iteration count (54 at step 1).
s_ildl = base;  s_ildl.k = 0;  s_ildl.iters = 312;
u_ildl = matvec_budget(s_ildl);
kbe = NaN;
for k = 1:500
    st = base;  st.k = k;  st.iters = 54;  st.build_E = true;
    if matvec_budget(st) > u_ildl, kbe = k; break; end
end
assert(isfinite(kbe) && kbe > 20 && kbe < 200, ...
       'T4: break-even k = %g, expected order 100', kbe);
fprintf(['  PASS T4: k=500 coarse apply %.1fx operator (measured 3.6x);\n' ...
         '          E rebuild alone %.0f units vs the whole ILDL budget of %.0f;\n' ...
         '          break-even at k ~ %d\n'], ratio, dE.coarse_build, u_ildl, kbe);
npass = npass + 1;

fprintf('=== test_matvec_budget: %d checks passed ===\n', npass);
