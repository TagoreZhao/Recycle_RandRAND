%TEST_CAPACITANCE  The capacitance matrix Cap = B + U' K_1^{-1} U is what it claims.
%
%   Run:  cd tests; test_capacitance
%
%   Cap is the only object in the scheme that is assembled by hand, and it is the
%   one whose conditioning the Woodbury forward error tracks.  Three things must
%   hold:
%
%     * it is SYMMETRIC (K_1^{-1} is symmetric and B is, so U'K_1^{-1}U must be).
%       woodbury_solve exploits this by reusing the precomputed Sel'K_1^{-1}Sel
%       block, so an asymmetry here would be a wiring error, not rounding;
%     * it equals an INDEPENDENTLY formed B + U'(K_1\U) that shares none of the
%       block-reuse shortcuts;
%     * its conditioning agrees with the number the prior art already reports for
%       exactly this matrix -- lowrank_update_basis's info.rcond_capacitance,
%       whose docstring names "a downstream Woodbury SOLVE" as the reason it is
%       computed.  This study is that downstream use, so the two must agree.
%
%   See also: woodbury_solve, lowrank_update_basis, seq_dCblk.

clear; clc;
thisFileDir = fileparts(mfilename('fullpath'));
studyDir    = fileparts(thisFileDir);
addpath(studyDir);
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);
fprintf('=== test_capacitance ===\n');

sopts = struct('case_name', 'bar_rotating', 'h0', 0.1, 'dt', 0.02, ...
               'Tstep', 61, 'nsteps', 5, 'verify', true, ...
               'use_cache', true, 'quiet', true);
S = build_stokes_sequence(sopts);
ctx = woodbury_context_init(S);
nC  = S.nC;
fprintf('  fixture: n=%d  nC=%d  Cap is %dx%d\n', S.n, nC, 2*nC, 2*nC);

np = 0;  nf = 0;

nprobe = 4;                                   % step 4 is well away from the ref
b = S.b{nprobe};
[~, info] = woodbury_solve(ctx, S, nprobe, b);
Cap = info.Cap;

% --- T1  symmetry --------------------------------------------------------
[np, nf] = chk(np, nf, ...
    sprintf('T1  Cap symmetric before symmetrization (%.2e < 1e-12)', ...
            info.cap_symres), ...
    info.cap_symres < 1e-12);

% --- T2  independent reassembly -----------------------------------------
% No block reuse, no cached Sel half: solve all 2nC columns fresh.
[U, dC] = seq_dCblk(S, nprobe, ctx.ref);
K1   = seq_K(S, ctx.ref);
Y0   = K1 \ full(U);
Bm   = [sparse(nC, nC), speye(nC); speye(nC), sparse(nC, nC)];
Cref = full(Bm) + full(U' * Y0);
Cref = (Cref + Cref') / 2;
relC = norm(Cap - Cref, 'fro') / max(norm(Cref, 'fro'), eps);
[np, nf] = chk(np, nf, ...
    sprintf('T2  Cap matches independent reassembly (%.2e < 1e-10)', relC), ...
    relC < 1e-10);

% --- T3  the size and shape are the ones the algebra predicts ------------
[np, nf] = chk(np, nf, ...
    sprintf('T3  Cap is 2nC x 2nC = %dx%d', 2*nC, 2*nC), ...
    isequal(size(Cap), [2*nC, 2*nC]));

% --- T4  conditioning agrees with the prior art -------------------------
% lowrank_update_basis computes rcond of this same Cap as a diagnostic for a
% downstream Woodbury solve.  Same matrix, different arithmetic path, so agree
% to within a factor of two rather than to rounding.
P_n = src.precond.make_ildl_precond(seq_K(S, nprobe), struct('mode', 'nofill'));
[~, lrinfo] = lowrank_update_basis(S, nprobe, P_n, [], ...
                                   struct('mode', 'invref', 'ref', ctx.ref));
ratio = info.cap_rcond / max(lrinfo.rcond_capacitance, realmin);
fprintf('    rcond: this study %.4e | lowrank_update_basis %.4e | ratio %.4f\n', ...
        info.cap_rcond, lrinfo.rcond_capacitance, ratio);
[np, nf] = chk(np, nf, ...
    sprintf('T4  rcond(Cap) agrees with lowrank_update_basis (ratio %.4f)', ratio), ...
    ratio > 0.5 && ratio < 2.0);

% --- T5  conditioning is finite and reported both ways ------------------
[np, nf] = chk(np, nf, ...
    sprintf('T5a cond(Cap) finite and >= 1 (%.3e)', info.cap_cond), ...
    isfinite(info.cap_cond) && info.cap_cond >= 1);
[np, nf] = chk(np, nf, ...
    sprintf('T5b smax/smin == cond (%.3e / %.3e)', info.cap_smax, info.cap_smin), ...
    abs(info.cap_smax / info.cap_smin - info.cap_cond) <= 1e-8 * info.cap_cond);

% --- T6  at the reference step the update collapses exactly -------------
% dC is identically zero there, so K_n == K_1 and the correction term must be
% SKIPPED, not merely small.  Asserted via correction_norm rather than against a
% standalone frozen solve: woodbury_solve batches the RHS in with the nC update
% columns, so its K_1^{-1}b differs in the last bits from a single-column solve.
b1 = S.b{ctx.ref};
[x1, info1] = woodbury_solve(ctx, S, ctx.ref, b1);
[np, nf] = chk(np, nf, ...
    'T6a dC is exactly zero at the reference step', ...
    info1.dC_is_zero && info1.dC_normF == 0);
[np, nf] = chk(np, nf, ...
    'T6b Woodbury correction is exactly zero at the reference step', ...
    info1.correction_norm == 0);
relRef = norm(x1 - woodbury_apply_ref(ctx, b1)) / max(norm(x1), eps);
[np, nf] = chk(np, nf, ...
    sprintf('T6c matches K_1^{-1}b to machine precision (%.2e < 1e-13)', relRef), ...
    relRef < 1e-13);

% --- T7  dC_rel is measured against the reference coupling block --------
[np, nf] = chk(np, nf, ...
    sprintf('T7  dC_rel == ||dC||_F/||Cblk_ref||_F (%.4f)', info.dC_rel), ...
    abs(info.dC_rel - norm(dC, 'fro') / norm(S.Cblk{ctx.ref}, 'fro')) < 1e-12);

fprintf('\n  %d passed, %d failed\n', np, nf);
if nf > 0
    error('test_capacitance:fail', '%d assertion(s) failed.', nf);
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
