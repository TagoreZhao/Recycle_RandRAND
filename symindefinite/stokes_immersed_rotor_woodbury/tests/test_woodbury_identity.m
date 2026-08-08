%TEST_WOODBURY_IDENTITY  The gate: does the Woodbury update actually solve K_n x = b?
%
%   Run:  cd tests; test_woodbury_identity
%
%   Everything in this study rests on one claim: a SINGLE exact factorization of
%   K_1, plus a rank-2nC capacitance correction, reproduces K_n \ b_n for every
%   later step.  If that fails, no figure downstream means anything.
%
%   THE FIXTURE IS A REAL SEQUENCE, not a synthetic perturbation.  It has to be:
%   as the Lagrange points move, pointLocation returns different host triangles,
%   so the coupling block's SPARSITY PATTERN changes -- and it is that pattern
%   change, not a smooth perturbation of fixed entries, that the low-rank form has
%   to absorb.  h0 = 0.1 keeps it to ~1600 unknowns and a few seconds.
%
%   T1 IS A NON-VACUITY GUARD.  If the uncorrected frozen inverse happened to be
%   accurate on this fixture, T2/T3 would pass without exercising the Woodbury
%   term at all.  T1 asserts the control is O(1) wrong first, so a pass on T2
%   means something.
%
%   See also: woodbury_solve, woodbury_context_init, build_stokes_sequence.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_woodbury_identity ===\n');

TOL_ERR    = 1e-9;      % forward error ||x_w - xref|| / ||xref||
TOL_RES    = 1e-10;     % true residual ||K_n x_w - b|| / ||b||
TOL_FROZEN = 1e-3;      % the control must be AT LEAST this wrong (non-vacuity)

% --- Fixture -------------------------------------------------------------
sopts = struct('case_name', 'bar_rotating', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', 5, 'verify', true, ...
               'use_cache', true, 'quiet', true);
S = build_stokes_sequence(sopts);
fprintf('  fixture: %s  n=%d  nC=%d  nsteps=%d\n', ...
        S.case_name, S.n, S.nC, S.nsteps);

ctx = woodbury_context_init(S);
fprintf('  context: ref=%d  nnz(L)=%d  fill=%.2f  setup %.3f s\n', ...
        ctx.ref, ctx.nnzL, ctx.fill_ratio, ctx.t_setup);

np = 0;  nf = 0;

% --- Per-step measurements ----------------------------------------------
err_w   = nan(S.nsteps, 1);
res_w   = nan(S.nsteps, 1);
err_f   = nan(S.nsteps, 1);
dC_rel  = nan(S.nsteps, 1);
nbs     = nan(S.nsteps, 1);
capcond = nan(S.nsteps, 1);

for n = 1:S.nsteps
    b    = S.b{n};
    xref = S.xref{n};
    Kn   = seq_K(S, n);

    [xw, info] = woodbury_solve(ctx, S, n, b);
    xf = woodbury_apply_ref(ctx, b);                       % the uncorrected control

    err_w(n)   = norm(xw - xref) / max(norm(xref), eps);
    res_w(n)   = norm(Kn * xw - b) / max(norm(b), eps);
    err_f(n)   = norm(xf - xref) / max(norm(xref), eps);
    dC_rel(n)  = info.dC_rel;
    nbs(n)     = info.n_backsolves;
    capcond(n) = info.cap_cond;

    fprintf(['    step %d  dC_rel %.3f  cond(Cap) %.2e | ' ...
             'woodbury err %.2e res %.2e | frozen err %.2e\n'], ...
            n, dC_rel(n), capcond(n), err_w(n), res_w(n), err_f(n));
end

% --- T1  non-vacuity: the control must be badly wrong, and dC must move --
later = 2:S.nsteps;
[np, nf] = chk(np, nf, ...
    sprintf('T1a frozen control is O(1) wrong by step>=2 (max %.2e > %.0e)', ...
            max(err_f(later)), TOL_FROZEN), ...
    max(err_f(later)) > TOL_FROZEN);
[np, nf] = chk(np, nf, ...
    sprintf('T1b coupling actually moves (min dC_rel %.3f > 0.01)', ...
            min(dC_rel(later))), ...
    min(dC_rel(later)) > 0.01);

% --- T2  forward error ---------------------------------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T2  woodbury forward error < %.0e (max %.2e)', ...
            TOL_ERR, max(err_w)), ...
    all(err_w < TOL_ERR));

% --- T3  true residual ---------------------------------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T3  woodbury true residual < %.0e (max %.2e)', ...
            TOL_RES, max(res_w)), ...
    all(res_w < TOL_RES));

% --- T4  at the reference step the update must be a no-op ----------------
[np, nf] = chk(np, nf, ...
    sprintf('T4  exact at the reference step (err %.2e < 1e-12)', ...
            err_w(ctx.ref)), ...
    err_w(ctx.ref) < 1e-12);

% --- T5  cost accounting: nC backsolves per step, not 2nC ----------------
% The Sel half of U = [dC, Sel] is time-independent, so K_1^{-1}Sel is solved
% once at setup.  If this ever reads 2*nC that optimization has regressed.
[np, nf] = chk(np, nf, ...
    sprintf('T5  nC (=%d) operator backsolves per step, not 2nC', S.nC), ...
    all(nbs == S.nC));

% --- T6  the correction is what fixes it, quantitatively -----------------
% Woodbury must beat the frozen control by many orders of magnitude, not by a
% constant factor -- a "fix" worth only 2x would mean the term is misscaled.
gain = median(err_f(later) ./ max(err_w(later), realmin));
[np, nf] = chk(np, nf, ...
    sprintf('T6  median error gain over frozen control %.1e > 1e6', gain), ...
    gain > 1e6);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_woodbury_identity:fail', '%d assertion(s) failed.', nf);
end

%==========================================================================
function [np, nf] = chk(np, nf, name, cond)
    if cond
        np = np + 1;
        fprintf('  PASS  %s\n', name);
    else
        nf = nf + 1;
        fprintf(2, '  FAIL  %s\n', name);
    end
end
