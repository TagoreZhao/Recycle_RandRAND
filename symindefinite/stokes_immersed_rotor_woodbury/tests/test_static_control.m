%TEST_STATIC_CONTROL  The falsification control: a sequence that does not move.
%
%   Run:  cd tests; test_static_control
%
%   disk_static holds its Lagrange points fixed, so the coupling block C(t) is
%   constant, dC is EXACTLY zero, and therefore K_n == K_1 exactly at every step.
%   Two things must follow, and both are stronger than "small":
%
%     * the Woodbury correction must be skipped, not merely tiny, so the returned
%       iterate equals the uncorrected frozen solve BIT-FOR-BIT;
%     * the frozen inverse must be accurate here.  That is what proves the
%       O(1) errors seen on bar_rotating come from the operator moving and not
%       from a bug in the frozen factorization or the ground truth.
%
%   A scheme that "works" on a moving sequence but fails this control is wired
%   wrong; a scheme that passes only this control has not been tested at all.
%
%   See also: test_woodbury_identity, woodbury_solve, define_motion_list.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_static_control ===\n');

sopts = struct('case_name', 'disk_static', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', 5, 'verify', true, ...
               'use_cache', true, 'quiet', true);
S = build_stokes_sequence(sopts);
ctx = woodbury_context_init(S);
fprintf('  fixture: %s  n=%d  nC=%d  nsteps=%d\n', ...
        S.case_name, S.n, S.nC, S.nsteps);

np = 0;  nf = 0;

all_zero   = true;
no_corr    = true;
max_err_w  = 0;
max_err_f  = 0;
max_dC_rel = 0;
max_wf     = 0;

for n = 1:S.nsteps
    b    = S.b{n};
    xref = S.xref{n};

    [xw, info] = woodbury_solve(ctx, S, n, b);
    xf = woodbury_apply_ref(ctx, b);

    all_zero   = all_zero && info.dC_is_zero && info.dC_normF == 0;
    % The sharp claim: the correction term was not merely small, it was SKIPPED.
    no_corr    = no_corr  && info.correction_norm == 0;
    max_err_w  = max(max_err_w, norm(xw - xref) / max(norm(xref), eps));
    max_err_f  = max(max_err_f, norm(xf - xref) / max(norm(xref), eps));
    max_dC_rel = max(max_dC_rel, info.dC_rel);
    max_wf     = max(max_wf, norm(xw - xf) / max(norm(xw), eps));
end

fprintf('  max dC_rel %.3e | woodbury err %.2e | frozen err %.2e | ||w-f|| %.2e\n', ...
        max_dC_rel, max_err_w, max_err_f, max_wf);

% --- T1  the premise: the operator genuinely does not move ---------------
[np, nf] = chk(np, nf, ...
    'T1  dC is EXACTLY zero at every step (nnz == 0, not just small)', ...
    all_zero && max_dC_rel == 0);

% --- T2  the correction term is skipped, exactly -------------------------
% Asserted on the solve's own iterate rather than against a separate frozen
% solve: woodbury_solve batches the RHS in with the nC update columns, so its
% K_1^{-1}b differs from a standalone single-column solve in the last bits
% (different BLAS blocking).  correction_norm == 0 is the claim that matters.
[np, nf] = chk(np, nf, ...
    'T2a Woodbury correction is EXACTLY zero (skipped, not small)', ...
    no_corr);
[np, nf] = chk(np, nf, ...
    sprintf('T2b woodbury == frozen to machine precision (%.2e < 1e-13)', max_wf), ...
    max_wf < 1e-13);

% --- T3  the frozen inverse is accurate when nothing moves ---------------
% This is the control that separates "the operator moved" from "the frozen
% factorization or the ground truth is broken".
[np, nf] = chk(np, nf, ...
    sprintf('T3  frozen inverse accurate on a static sequence (%.2e < 1e-10)', ...
            max_err_f), ...
    max_err_f < 1e-10);

[np, nf] = chk(np, nf, ...
    sprintf('T4  woodbury equally accurate (%.2e < 1e-10)', max_err_w), ...
    max_err_w < 1e-10);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_static_control:fail', '%d assertion(s) failed.', nf);
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
