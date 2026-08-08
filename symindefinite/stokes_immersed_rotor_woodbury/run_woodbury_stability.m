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
fprintf('%5s %11s %11s %11s %11s %11s %11s\n', 'step', 'cond(K_n)', ...
        'cond(Cap)', 'rho=|y|/|x|', 'wood_err', 'kCap*eps', 'kKn*eps');
for n = PROBE
    b  = S.b{n};
    Kn = seq_K(S, n);
    [xw, info] = woodbury_solve(ctx, S, n, b);
    y    = woodbury_apply_ref(ctx, b);
    xr   = S.xref{n};
    kKn  = condest(Kn);
    fprintf('%5d %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e\n', n, kKn, ...
            info.cap_cond, norm(y)/norm(xw), ...
            norm(xw - xr)/norm(xr), info.cap_cond*eps, kKn*eps);
end
fprintf(['\nkappa(K_n)*eps overstates the observed error by ~5 orders of magnitude:\n' ...
         'the update is NOT governed by the operator''s conditioning.\n']);

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

b     = Kn * v;
xtrue = v;                                  % exact, by construction
[xw, info] = woodbury_solve(ctx, S, NSTEP, b);
xbs   = Kn \ b;
y     = woodbury_apply_ref(ctx, b);
rho   = norm(y) / norm(xtrue);

fprintf('  realized rho              = %.3e\n', rho);
fprintf('  woodbury  abs rel err     = %.3e\n', norm(xw  - xtrue)/norm(xtrue));
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
