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

% --- T1  symmetry, measured but NOT imposed ------------------------------
% Cap is symmetric in exact arithmetic, and woodbury_solve deliberately does not
% symmetrize it -- that would be free accuracy handed to a routine whose accuracy
% is the thing under study.  So cap_symres must be nonzero (the repair is really
% gone) and at rounding level (the wiring is really right).
[np, nf] = chk(np, nf, ...
    sprintf('T1  Cap symmetric to rounding, left unrepaired (%.2e < 1e-12)', ...
            info.cap_symres), ...
    info.cap_symres > 0 && info.cap_symres < 1e-12);

% --- T2  independent reassembly -----------------------------------------
% No block reuse, no cached Sel half, and no symmetrization: solve all 2nC columns
% fresh and form C^{-1} + U'Y0 exactly as the identity is written.
[U, dC] = seq_dCblk(S, nprobe, ctx.ref);
K1   = seq_K(S, ctx.ref);
Y0   = K1 \ full(U);
Bm   = [sparse(nC, nC), speye(nC); speye(nC), sparse(nC, nC)];
Cref = full(Bm) + full(U' * Y0);
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

% --- T6  at the reference step the update collapses ---------------------
% dC is identically zero there, so K_n == K_1 and the correction is provably zero.
% It is NOT special-cased away -- the naive path computes and rounds it like any
% other step -- so this asserts a bound, not an identity.  In practice it comes out
% exactly 0, and structurally so: with dC = 0 the first block row of Cap is [0 I]
% against a zero right-hand side, which forces the Sel half of w to zero, while the
% dC half of Y0 is the exact zero matrix.  Printed so a regression to ~1e-16 is
% visible rather than silently absorbed by the tolerance.
b1 = S.b{ctx.ref};
[x1, info1] = woodbury_solve(ctx, S, ctx.ref, b1);
[np, nf] = chk(np, nf, ...
    'T6a dC is exactly zero at the reference step', ...
    info1.dC_is_zero && info1.dC_normF == 0);
[np, nf] = chk(np, nf, ...
    sprintf('T6b Woodbury correction vanishes at the reference step (%.2e < 1e-13)', ...
            info1.correction_norm), ...
    info1.correction_norm < 1e-13);
relRef = norm(x1 - woodbury_apply_ref(ctx, b1)) / max(norm(x1), eps);
[np, nf] = chk(np, nf, ...
    sprintf('T6c matches K_1^{-1}b to machine precision (%.2e < 1e-13)', relRef), ...
    relRef < 1e-13);

% --- T7  dC_rel is measured against the reference coupling block --------
[np, nf] = chk(np, nf, ...
    sprintf('T7  dC_rel == ||dC||_F/||Cblk_ref||_F (%.4f)', info.dC_rel), ...
    abs(info.dC_rel - norm(dC, 'fro') / norm(S.Cblk{ctx.ref}, 'fro')) < 1e-12);

% --- T8  U is ALREADY orthogonal between its two blocks -----------------
% dC occupies only the velocity rows ([Cu;0;0]) and Sel only the multiplier rows
% ([0;0;I]) -- disjoint support, so dC'*Sel is EXACTLY zero.  This is why
% woodbury_solve does not orthogonalize U: there is no cross-block conditioning
% to fix, and dC is well conditioned on its own.
[np, nf] = chk(np, nf, ...
    sprintf('T8a dC and Sel are exactly orthogonal (||dC''Sel|| = %.1e)', ...
            norm(full(dC' * ctx.Sel))), ...
    norm(full(dC' * ctx.Sel)) == 0);
[np, nf] = chk(np, nf, ...
    sprintf('T8b dC is well conditioned on its own (cond = %.2f < 1e3)', ...
            cond(full(dC))), ...
    cond(full(dC)) < 1e3);

% --- T9  Cap survives a RANK-DEFICIENT dC ------------------------------
% B = [0 I; I 0] couples the two blocks, so even if dC*v = 0 then
% Cap*[v;0] = [0;v] ~= 0: Cap is nonsingular whatever dC's rank.  That robustness
% is the reason for keeping B (its own inverse, kappa = 1) rather than
% orthogonalizing dC = Qd*Rd, which would force an inversion of Rd and die here.
% If anyone "cleans up" woodbury_solve by orthogonalizing U, this test says why not.
dCr = dC;
dCr(:, end) = dCr(:, 1);                  % exact duplicate => rank nC-1
Ydc  = woodbury_apply_ref(ctx, full(dCr));
Icc  = eye(nC);
CapR = [full(dCr' * Ydc),            full(dCr' * ctx.YSel) + Icc; ...
        full(ctx.Sel' * Ydc) + Icc,  ctx.SelYSel];
CapR = (CapR + CapR') / 2;
[~, Rd] = qr(full(dCr), 0);
fprintf('    rank-deficient dC: rank %d/%d | cond(Cap) %.3e | cond(Rd) %.3e\n', ...
        rank(full(dCr)), nC, cond(CapR), cond(Rd));
[np, nf] = chk(np, nf, ...
    sprintf('T9a Cap stays well conditioned on rank-deficient dC (%.3e < 1e6)', ...
            cond(CapR)), ...
    cond(CapR) < 1e6);
[np, nf] = chk(np, nf, ...
    sprintf('T9b the orthogonalized alternative would break (cond(Rd) %.1e > 1e12)', ...
            cond(Rd)), ...
    cond(Rd) > 1e12);

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
