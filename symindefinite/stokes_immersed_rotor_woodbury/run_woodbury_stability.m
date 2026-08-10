%RUN_WOODBURY_STABILITY  Why is the Woodbury error 1e-14 when SMW is famously unstable?
%
%   Run:  run_woodbury_stability
%
%   The main benchmark reports a forward error of ~1e-14 while kappa(K_n) ~ 3e6, so a
%   kappa(K)-amplified error would be ~1e-9.  This script measures the quantities the
%   Sherman-Morrison-Woodbury error analysis actually depends on, and produces the
%   numbers quoted in README section 4.1.  It writes no files.
%
%   PART 1 -- the three candidate error scales, per step:
%     kappa(K_n)*eps      what the error would be if it were governed by the operator
%     kappa(Cap)*eps      the small dense solve inside the update
%     rho = ||y||/||x||   the CANCELLATION ratio, y = K_1^{-1}b: the formula subtracts
%                         two vectors, so the error scales with how much larger the
%                         subtrahend is than the result
%
%   PART 1b -- the full metric set the brief asks for, on the physical sequence:
%   forward error, true residual, normwise backward error, and BOTH cancellation
%   factors (cancel_cap for the small matrix Cap, cancel_sub for the final
%   subtraction).  run_woodbury_scalar_stress runs the same instruments on two
%   constructed systems where the method loses every digit at kappa = 1; this part
%   is the negative control, showing the same mechanism present but unexcited.
%
%   PART 2 -- trigger the instability on purpose.  rho is maximized over b by taking
%   b = K_n*v with v the leading singular vector of K_1^{-1}K_n, since then
%
%       x = K_n^{-1}b = v   EXACTLY (no solve, so the reference is not itself a solve)
%       y = K_1^{-1}b = (K_1^{-1}K_n) v,   ||y||/||x|| = ||K_1^{-1}K_n||_2.
%
%   v is found by power iteration on (K_1^{-1}K_n)'(K_1^{-1}K_n).  This is the one place
%   in the study with an exactly known solution, so it is also the only clean measurement
%   of ABSOLUTE forward error -- everywhere else the reference is itself a direct solve.
%
%   WHY kappa(K_n) DOES NOT ENTER.  y and Y_0 = K_1^{-1}U are applied with the SAME
%   factors, so a consistently perturbed (K_1 + dK_1)^{-1} makes the formula return
%   exactly (K_n + dK_1)^{-1}b: the rounding is a BACKWARD perturbation of K_n, not an
%   amplified forward error.  Refreshing the factors between Y_0 and y would forfeit this.
%
%   See also: woodbury_solve, woodbury_apply_ref, run_woodbury_benchmark.

clear; clc;
add_woodbury_paths();
assert_woodbury_helpers();

rng(0);

CASE  = 'bar_rotating';
PROBE = [2 10 20 31 40 60];
NSTEP = 20;                 % the step used for the adversarial experiment
NPOW  = 60;                 % power iterations for the leading singular vector

params = make_woodbury_params();
S = build_stokes_sequence(struct('case_name', CASE, 'h0', params.h0, ...
        'dt', params.dt, 'Tstep', params.Tstep, 'nsteps', params.Tstep - 1, ...
        'verify', false, 'use_cache', true, 'quiet', true));
ctx = woodbury_context_init(S);
K1  = seq_K(S, ctx.ref);

fprintf('=== woodbury stability: %s, n = %d, nC = %d ===\n', CASE, S.n, S.nC);
fprintf('condest(K_1) = %.3e     eps = %.3e\n\n', condest(K1), eps);

% ---- PART 1: which condition number does the error follow? ---------------
np = numel(PROBE);
M  = struct('kKn', nan(np,1), 'kCap', nan(np,1), 'rho', nan(np,1), ...
            'fwd', nan(np,1), 'res', nan(np,1), 'bwd', nan(np,1), ...
            'cCap', nan(np,1), 'cSub', nan(np,1));

fprintf('%5s %11s %11s %11s %11s %11s %11s\n', 'step', 'cond(K_n)', ...
        'cond(Cap)', 'rho=|y|/|x|', 'wood_err', 'kCap*eps', 'kKn*eps');
for i = 1:np
    n  = PROBE(i);
    b  = S.b{n};
    Kn = seq_K(S, n);
    [xw, info] = woodbury_solve(ctx, S, n, b);
    xr = S.xref{n};
    r  = b - Kn * xw;

    M.kKn(i)  = condest(Kn);
    M.kCap(i) = info.cap_cond;
    M.rho(i)  = info.rho;
    M.fwd(i)  = norm(xw - xr) / norm(xr);
    M.res(i)  = norm(r) / norm(b);
    M.bwd(i)  = norm(r) / (norm(Kn, 'fro') * norm(xw) + norm(b));
    M.cCap(i) = info.cancel_cap;
    M.cSub(i) = info.cancel_sub;

    fprintf('%5d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e\n', n, M.kKn(i), ...
            M.kCap(i), M.rho(i), M.fwd(i), M.kCap(i)*eps, M.kKn(i)*eps);
end
fprintf(['\nkappa(K_n)*eps overstates the observed error by ~5 orders of magnitude:\n' ...
         'the update is NOT governed by the operator''s conditioning.\n']);

% ---- PART 1b: the full metric set, including what DOES govern it --------
% run_woodbury_scalar_stress shows that the two cancellation factors, not any
% condition number, predict the Woodbury error on constructed systems where the
% method fails outright.  This is the same measurement on the physical sequence,
% where it does not -- so the two tables are read together: the mechanism is
% present and instrumented here, it is simply not excited.
fprintf('\n=== the full metric set on the physical sequence ===\n');
fprintf('%5s %11s %11s %11s %11s %11s %11s\n', 'step', 'fwd_err', ...
        'residual', 'bwd_err', 'cancel_cap', 'cancel_sub', 'cSub*eps');
for i = 1:np
    fprintf('%5d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e\n', PROBE(i), ...
            M.fwd(i), M.res(i), M.bwd(i), M.cCap(i), M.cSub(i), M.cSub(i)*eps);
end
fprintf(['\ncancel_sub in [%.2f, %.2f]: the final subtraction is inert -- the correction\n' ...
         'is smaller than the answer it corrects, so no leading digits annihilate.\n' ...
         'cancel_cap in [%.0f, %.0f]: NOT 1, but two digits, not fourteen.  It is the\n' ...
         'binding predictor here -- cancel_cap*eps = %.1e brackets the observed errors\n' ...
         '(%.1e to %.1e) while cancel_sub*eps = %.1e sits below them.\n'], ...
        min(M.cSub), max(M.cSub), min(M.cCap), max(M.cCap), ...
        max(M.cCap)*eps, min(M.fwd), max(M.fwd), max(M.cSub)*eps);
fprintf(['\nCAVEAT: cond(Cap) is also ~1e2 on this sequence, so these six rows cannot\n' ...
         'separate cancel_cap*eps from cond(Cap)*eps as the governing scale.  What they\n' ...
         'do settle is that neither is kappa(K_n)*eps ~ 6e-10.  The constructed families\n' ...
         'in run_woodbury_scalar_stress DO separate them: there cond(S) stays at 1 while\n' ...
         'cancel_S reaches 4e15, and the error follows cancel_S.\n']);
fprintf(['\nBackward error, max %.1e: the computed iterate exactly solves a system within\n' ...
         'rounding of K_n.  Woodbury is backward stable here -- which is the claim the\n' ...
         'alpha = 1e16 family falsifies in general, reaching a backward error of 1.0.\n'], ...
        max(M.bwd));

% ---- PART 2: force cancellation and watch it break ----------------------
fprintf('\n=== adversarial RHS at step %d: maximize rho ===\n', NSTEP);
Kn      = seq_K(S, NSTEP);
applyM  = @(v) woodbury_apply_ref(ctx, Kn * v);        % K_1^{-1} K_n
applyMt = @(v) Kn' * woodbury_apply_ref(ctx, v);       % its transpose

v = randn(S.n, 1);  v = v / norm(v);
for it = 1:NPOW
    w = applyMt(applyM(v));
    v = w / norm(w);
end
fprintf('||K_1^{-1}K_n||_2 = max_b rho = %.3e\n', norm(applyM(v)));

% Kept under _adv names: PART 3 reuses them after its own loop overwrites b/xw.
b_adv = Kn * v;
xtrue = v;                                  % exact, by construction
[xw_adv, info] = woodbury_solve(ctx, S, NSTEP, b_adv);
xbs   = Kn \ b_adv;
y     = woodbury_apply_ref(ctx, b_adv);
rho   = norm(y) / norm(xtrue);

fprintf('  realized rho              = %.3e\n', rho);
fprintf('  woodbury  abs rel err     = %.3e\n', norm(xw_adv - xtrue)/norm(xtrue));
fprintf('  backslash abs rel err     = %.3e   <-- unmoved\n', ...
        norm(xbs - xtrue)/norm(xtrue));
fprintf('  rho * kappa(Cap) * eps    = %.3e   (upper bound)\n', ...
        rho * info.cap_cond * eps);

% ---- the benign contrast, same step, same matrices ----------------------
b2 = S.b{NSTEP};
xw2 = woodbury_solve(ctx, S, NSTEP, b2);
y2  = woodbury_apply_ref(ctx, b2);
fprintf('\n  physical RHS, same step: rho = %.3f, err = %.3e\n', ...
        norm(y2)/norm(xw2), norm(xw2 - S.xref{NSTEP})/norm(S.xref{NSTEP}));
fprintf(['\nSMW''s instability is REAL on this operator -- it is reachable with a\n' ...
         'right-hand side chosen to cancel.  The rotor physics never asks for one:\n' ...
         'the correction stays smaller than the solution it corrects (rho <= 1.01).\n']);

% ---- PART 3: would ORTHOGONALIZING U help? ------------------------------
% The usual instinct for low-rank updates is to orthogonalize the factors.  Here
% it is measured to be neutral at best, for three reasons this part demonstrates:
%
%   (a) U's two blocks are ALREADY exactly orthogonal.  dC = [Cu;0;0] lives in the
%       velocity rows, Sel = [0;0;I] in the multiplier rows -- disjoint support --
%       and dC is well conditioned on its own.  Nothing to normalize.
%   (b) rho is BASIS-INDEPENDENT: it involves only b, K_1 and K_n, not U.  So no
%       reparametrization of U can touch the dominant error term.
%   (c) orthogonalizing dC = Qd*Rd turns the middle matrix into
%       Btil = [0 Rd; Rd' 0], whose inverse needs Rd^{-1}.  The current
%       B = [0 I; I 0] IS its own inverse (kappa = 1) and is never inverted, which
%       is what makes Cap survive a rank-deficient dC (see tests T9a/T9b).
fprintf('\n=== would orthogonalizing U = [dC, Sel] help? ===\n');
fprintf('%5s %11s %11s %11s %11s %11s\n', 'step', 'cond(Cap)', 'cond(CapQ)', ...
        'err_orig', 'err_orth', 'cond(Rd)');
for n = PROBE
    b = S.b{n};  xr = S.xref{n};
    [xw, info] = woodbury_solve(ctx, S, n, b);
    [xo, capo, cRd] = local_orth_variant(ctx, S, n, b);
    fprintf('%5d %11.3e %11.3e %11.3e %11.3e %11.3e\n', n, info.cap_cond, capo, ...
            norm(xw - xr)/norm(xr), norm(xo - xr)/norm(xr), cRd);
end

[xo_adv, capo] = local_orth_variant(ctx, S, NSTEP, b_adv);
fprintf(['\nand on the adversarial RHS, where the error actually IS bad:\n' ...
         '  original %.3e   orthogonalized %.3e  (cond(CapQ) %.3e)\n'], ...
        norm(xw_adv - xtrue)/norm(xtrue), norm(xo_adv - xtrue)/norm(xtrue), capo);
fprintf(['Orthogonalization cannot help: rho does not involve U at all.\n' ...
         'See tests/test_capacitance.m T8-T9 for the structural argument.\n']);

%==========================================================================
function [x, capcond, cRd] = local_orth_variant(ctx, S, n, b)
%LOCAL_ORTH_VARIANT  The orthogonalized formulation, for comparison only.
%   dC = Qd*Rd, so U = [Qd, Sel] with Btil = [0 Rd; Rd' 0] and
%   Btil^{-1} = [0 Rd^{-T}; Rd^{-1} 0] -- note the required inversion of Rd.
    nC = ctx.nC;
    [~, dC]  = seq_dCblk(S, n, ctx.ref);
    [Qd, Rd] = qr(full(dC), 0);
    Z  = woodbury_apply_ref(ctx, [Qd, b]);
    YQ = Z(:, 1:nC);   y = Z(:, nC + 1);
    Ri = Rd \ eye(nC);
    Cap = [full(Qd' * YQ),            full(Qd' * ctx.YSel) + Ri'; ...
           full(ctx.Sel' * YQ) + Ri,  ctx.SelYSel];
    Cap     = (Cap + Cap') / 2;
    capcond = cond(Cap);
    cRd     = cond(Rd);
    w = Cap \ [Qd' * y; full(ctx.Sel' * y)];
    x = y - YQ * w(1:nC) - ctx.YSel * w(nC+1:end);
end
