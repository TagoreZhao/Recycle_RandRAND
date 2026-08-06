% TEST_BASELINE_DRIFT  Is there a drift story to tell at all?
%
% The whole study rests on the frozen exact inverse going stale as S(t) moves.
% If chol(S_1) stayed a 1-iteration preconditioner forever there would be
% nothing to recycle against.  This runs the BASELINES ONLY (no deflation) and
% checks that the frozen arm degrades, that the frozen inverse itself is what
% drifts, and that it is nonetheless still worth having.
%
% Run:  cd tests; test_baseline_drift

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisFileDir));
add_schur_paths();
assert_local_helpers();
rng(1);

params = make_schur_params();
params.h0     = 0.1;
params.Tstep  = 13;                     % 12 solves -- enough to see the trend
params.standalone_variants = [];        % baselines only
params.COMPUTE_SPECTRUM    = true;

[cfg, ~] = schur_make_cfg('bar_rotating', params, []);
Astat = solve_schur_sequence(cfg, params, '');

its_chol  = Astat.solver_its.chol;
its_unp   = Astat.solver_its.pcg_unprec;

fprintf('\n  step | unprec   chol |   kappa    ReldiffF  InvRelDiff\n');
for n = 1:numel(its_chol)
    fprintf('  %4d | %6d %6d | %9.3e %9.3e %9.3e\n', ...
            n, its_unp(n), its_chol(n), ...
            Astat.kappa(n), Astat.ReldiffF(n), Astat.InvRelDiff(n));
end

npass = 0;

% T1: everything converged
for k = {'pcg_unprec', 'chol'}
    fl = Astat.solver_flag.(k{1});
    assert(all(fl == 0), 'T1: arm %s failed to converge (flags %s)', ...
           k{1}, mat2str(unique(fl)'));
    npass = npass + 1;
end

% T2: the Schur route reproduces K\b at every step
assert(max(Astat.vel_recovery_err) < 1e-9, ...
    'T2: velocity recovery error %.3e too large', max(Astat.vel_recovery_err));
npass = npass + 1;

% T3: the frozen chol is exact on the matrix it was built from
assert(its_chol(1) <= 2, ...
    'T3: frozen chol took %d iterations on S_1 (expected <= 2)', its_chol(1));
npass = npass + 1;

% T4: it degrades as S drifts -- this is the drift story
assert(max(its_chol) > its_chol(1), ...
    'T4: frozen chol never degraded (max %d vs first %d) -- no drift to study', ...
    max(its_chol), its_chol(1));
npass = npass + 1;

% T5: the MECHANISM, not just the symptom -- the frozen inverse itself is stale.
% If iterations grew while InvRelDiff stayed at zero, the degradation would be
% coming from somewhere other than staleness.
%
% NOT ASSERTED AS MONOTONE GROWTH.  The bar sweeps two full revolutions over
% [0, Tmax], so S(t) moves AWAY FROM AND BACK TOWARD S_1 rather than drifting
% off in one direction; InvRelDiff oscillates (measured: 0.145 at step 2, 0.086
% at step 12).  What matters is that it is never small after step 1.
assert(Astat.InvRelDiff(1) < 1e-10, ...
    'T5a: the frozen inverse is not exact on the matrix it was built from (InvRelDiff %.3e)', ...
    Astat.InvRelDiff(1));
assert(min(Astat.InvRelDiff(2:end)) > 1e-2, ...
    'T5b: the frozen inverse stayed accurate (min InvRelDiff %.3e) -- nothing has gone stale', ...
    min(Astat.InvRelDiff(2:end)));
npass = npass + 2;

% T6: the baseline is still worth having -- a stale exact factor must beat no
% preconditioner at all, or there is nothing to improve on.
assert(median(its_chol) < median(its_unp), ...
    'T6: frozen chol (%g) did not beat unpreconditioned PCG (%g)', ...
    median(its_chol), median(its_unp));
npass = npass + 1;

% T7: the operator really is moving
assert(median(Astat.ReldiffF(2:end)) > 1e-6, ...
    'T7: S barely changes between steps (median ReldiffF %.3e)', ...
    median(Astat.ReldiffF(2:end)));
npass = npass + 1;

fprintf('\ntest_baseline_drift: ALL %d ASSERTIONS PASSED\n', npass);
fprintf('  frozen chol: %d -> %d iterations (factor %.1f)\n', ...
        its_chol(1), its_chol(end), its_chol(end) / max(its_chol(1), 1));
fprintf('  unprec %d-%d\n', min(its_unp), max(its_unp));
fprintf('  InvRelDiff: %.3e -> %.3e\n', Astat.InvRelDiff(2), Astat.InvRelDiff(end));
fprintf('  kappa(S): %.3e -> %.3e\n', Astat.kappa(1), Astat.kappa(end));
