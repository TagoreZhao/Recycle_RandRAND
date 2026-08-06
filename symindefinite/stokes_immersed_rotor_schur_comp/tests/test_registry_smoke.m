% TEST_REGISTRY_SMOKE  End-to-end run with every arm enabled.
%
% Exercises the four-arm registry on a short sequence: every arm must converge
% to the true solution, and the deflation arms must build a coarse space of the
% requested width and then reuse it for the rest of the sequence.
%
% DELIBERATELY WEAK PERFORMANCE BOUNDS.  Whether deflation actually pays off on
% this operator is the question the benchmark exists to answer, so asserting a
% win here would be asserting the answer.  (The old wording justified weak
% bounds by "after ichol there is essentially no low-eigenvalue cluster left" --
% that measurement was made on the ichol-preconditioned operator and says
% nothing about the raw S this study now deflates.  Re-measure with
% run_schur_spectrum before drawing any conclusion.)  The only iteration-count
% assertions are against the unpreconditioned baseline.
%
% Run:  cd tests; test_registry_smoke

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
addpath(fileparts(thisFileDir));
add_schur_paths();
assert_local_helpers();
rng(1);

params = make_schur_params();
params.h0    = 0.1;
params.Tstep = 9;                       % 8 solves
% nS ~ 542 on this coarse mesh, so scale the coarse space down with it
params.sm_eig           = 30;
params.COMPUTE_SPECTRUM = true;

[cfg, ~] = schur_make_cfg('bar_rotating', params, []);
Astat = solve_schur_sequence(cfg, params, '');

keys = Astat.solver_keys;
fprintf('\n  step |');
fprintf(' %16s', keys{:});
fprintf('\n');
for n = 1:Astat.nsteps
    fprintf('  %4d |', n);
    for i = 1:numel(keys)
        fprintf(' %16d', Astat.solver_its.(keys{i})(n));
    end
    fprintf('\n');
end

npass = 0;

% T1: every arm converged at every step
for i = 1:numel(keys)
    fl = Astat.solver_flag.(keys{i});
    assert(all(fl == 0), 'T1: arm %s failed (flags %s)', ...
           keys{i}, mat2str(unique(fl)'));
    npass = npass + 1;
end

% T2: every arm reached the true solution
for i = 1:numel(keys)
    er = Astat.solver_err.(keys{i});
    assert(max(er) < 1e-5, 'T2: arm %s error %.3e too large', keys{i}, max(er));
    npass = npass + 1;
end

% T3: the registry is exactly the four expected arms, in reporting order
expect = {'pcg_unprec', 'chol', 'deflate_exact', 'deflate_gaussian'};
assert(isequal(keys(:)', expect), 'T3: registry is %s, expected %s', ...
       strjoin(keys, ','), strjoin(expect, ','));
npass = npass + 1;

% T4: no proxy-preconditioner or Krylov-recycling bookkeeping survives on Astat.
% ichol_shift in particular would mean a sparse proxy is still being built.
for f = {'ichol_shift', 'krylov_build_its', 'krylov_freeze_step', 'krylov_freeze_dim'}
    assert(~isfield(Astat, f{1}), ...
           'T4: removed field %s is still populated', f{1});
    npass = npass + 1;
end

% T5: both coarse spaces were built, at the requested width.  A width below
% sm_eig means `orth` found the sketch rank deficient -- legitimate, but it must
% be recorded, which is what deflat_dim exists for.
for key = {'deflate_exact', 'deflate_gaussian'}
    assert(isfield(Astat.deflat_dim, key{1}), ...
           'T5a: no coarse-space width recorded for %s', key{1});
    m = Astat.deflat_dim.(key{1});
    assert(m >= 1 && m <= params.sm_eig, ...
           'T5b: %s coarse width %d outside [1, %d]', key{1}, m, params.sm_eig);
    npass = npass + 2;
end
if Astat.deflat_dim.deflate_gaussian < params.sm_eig
    warning('test_registry_smoke:sketchRankDrop', ...
        'the Gaussian sketch realized %d of %d columns', ...
        Astat.deflat_dim.deflate_gaussian, params.sm_eig);
end

% T6: deflation is not actively harmful -- both arms beat no preconditioner.
% This is the weakest bound that would still catch a wrongly built projector
% (a P on the wrong operator makes the deflated solve far worse than plain PCG).
for key = {'deflate_exact', 'deflate_gaussian'}
    assert(median(Astat.solver_its.(key{1})) < median(Astat.solver_its.pcg_unprec), ...
        'T6: %s (%g) did not beat unpreconditioned PCG (%g)', key{1}, ...
        median(Astat.solver_its.(key{1})), median(Astat.solver_its.pcg_unprec));
    npass = npass + 1;
end

% T7: the coarse space is built ONCE and recycled.  With DEFLAT_PREC_REFRESH
% huge, a rebuild at any later step would be a silent cadence bug; the frozen
% chol it is built from is likewise built exactly once.
assert(isscalar(Astat.chol_built_step) && Astat.chol_built_step == 1, ...
    'T7: the frozen chol was built at steps %s, expected step 1 only', ...
    mat2str(Astat.chol_built_step));
npass = npass + 1;

% T8: tau was resolved to lambda_max(S_1), not left empty.  Captured modes land
% exactly at tau, so a tau below the top of the spectrum would relocate them
% rather than remove them.
assert(isfinite(Astat.tau) && Astat.tau > 0, ...
    'T8: tau was not resolved (got %s)', mat2str(Astat.tau));
assert(abs(Astat.tau - Astat.lambda_max(1)) < 1e-10 * Astat.lambda_max(1), ...
    'T8: tau %.6e != lambda_max(S_1) %.6e', Astat.tau, Astat.lambda_max(1));
npass = npass + 2;

fprintf('\ntest_registry_smoke: ALL %d ASSERTIONS PASSED\n', npass);
fprintf('  median iterations:');
for i = 1:numel(keys)
    fprintf('  %s=%g', keys{i}, median(Astat.solver_its.(keys{i})));
end
fprintf('\n  coarse-space width: exact=%d, gaussian=%d (requested %d)\n', ...
        Astat.deflat_dim.deflate_exact, ...
        Astat.deflat_dim.deflate_gaussian, params.sm_eig);
fprintf('  tau = %.6e (= lambda_max(S_1))\n', Astat.tau);
